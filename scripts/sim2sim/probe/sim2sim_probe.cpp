// Sim2sim recorder: logs rt/lowstate, rt/lowcmd and rt/sportmodestate with arrival times.
//
// Usage: sim2sim_probe <out_dir> <seconds> [network=lo] [domain=0]
// Writes lowstate.csv, lowcmd.csv, sport.csv into <out_dir>; times are CLOCK_MONOTONIC in ns,
// so they line up with gamepad.py's events.log (time.monotonic_ns()).
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

#include <array>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

constexpr int kJoints = 29;

static int64_t now_ns()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

struct LowStateRow
{
    int64_t t;
    uint32_t tick;
    float q[kJoints], dq[kJoints], tau[kJoints];
    float quat[4], gyro[3];
    uint16_t buttons;
    float lx, rx, ry, l2, ly;
};

struct LowCmdRow
{
    int64_t t;
    float q[kJoints];
    float kp0, kd0, kp15;
};

struct SportRow
{
    int64_t t;
    float pos[3], vel[3];
};

static std::atomic<bool> stop_flag{false};
static std::mutex ls_mutex, lc_mutex, sp_mutex;
static std::vector<LowStateRow> ls_rows;
static std::vector<LowCmdRow> lc_rows;
static std::vector<SportRow> sp_rows;

static void on_lowstate(const void *message)
{
    const auto &m = *static_cast<const unitree_hg::msg::dds_::LowState_ *>(message);
    LowStateRow r;
    r.t = now_ns();
    r.tick = m.tick();
    for (int i = 0; i < kJoints; ++i)
    {
        r.q[i] = m.motor_state()[i].q();
        r.dq[i] = m.motor_state()[i].dq();
        r.tau[i] = m.motor_state()[i].tau_est();
    }
    for (int i = 0; i < 4; ++i) r.quat[i] = m.imu_state().quaternion()[i];
    for (int i = 0; i < 3; ++i) r.gyro[i] = m.imu_state().gyroscope()[i];
    // wireless_remote: head[2], buttons (uint16), lx, rx, ry, L2, ly (float32), see unitree_joystick.hpp
    const uint8_t *w = m.wireless_remote().data();
    std::memcpy(&r.buttons, w + 2, 2);
    std::memcpy(&r.lx, w + 4, 4);
    std::memcpy(&r.rx, w + 8, 4);
    std::memcpy(&r.ry, w + 12, 4);
    std::memcpy(&r.l2, w + 16, 4);
    std::memcpy(&r.ly, w + 20, 4);
    std::lock_guard<std::mutex> lock(ls_mutex);
    ls_rows.push_back(r);
}

static void on_lowcmd(const void *message)
{
    const auto &m = *static_cast<const unitree_hg::msg::dds_::LowCmd_ *>(message);
    LowCmdRow r;
    r.t = now_ns();
    for (int i = 0; i < kJoints; ++i) r.q[i] = m.motor_cmd()[i].q();
    r.kp0 = m.motor_cmd()[0].kp();
    r.kd0 = m.motor_cmd()[0].kd();
    r.kp15 = m.motor_cmd()[15].kp();
    std::lock_guard<std::mutex> lock(lc_mutex);
    lc_rows.push_back(r);
}

static void on_sport(const void *message)
{
    const auto &m = *static_cast<const unitree_go::msg::dds_::SportModeState_ *>(message);
    SportRow r;
    r.t = now_ns();
    for (int i = 0; i < 3; ++i)
    {
        r.pos[i] = m.position()[i];
        r.vel[i] = m.velocity()[i];
    }
    std::lock_guard<std::mutex> lock(sp_mutex);
    sp_rows.push_back(r);
}

static void write_csv(const std::string &dir)
{
    FILE *f = std::fopen((dir + "/lowstate.csv").c_str(), "w");
    std::fprintf(f, "t_ns,tick");
    for (const char *name : {"q", "dq", "tau"})
        for (int i = 0; i < kJoints; ++i) std::fprintf(f, ",%s%d", name, i);
    std::fprintf(f, ",qw,qx,qy,qz,gx,gy,gz,buttons,lx,rx,ry,l2,ly\n");
    for (const auto &r : ls_rows)
    {
        std::fprintf(f, "%lld,%u", (long long)r.t, r.tick);
        for (float v : r.q) std::fprintf(f, ",%.9g", v);
        for (float v : r.dq) std::fprintf(f, ",%.9g", v);
        for (float v : r.tau) std::fprintf(f, ",%.9g", v);
        for (float v : r.quat) std::fprintf(f, ",%.9g", v);
        for (float v : r.gyro) std::fprintf(f, ",%.9g", v);
        std::fprintf(f, ",%u,%.9g,%.9g,%.9g,%.9g,%.9g\n", r.buttons, r.lx, r.rx, r.ry, r.l2, r.ly);
    }
    std::fclose(f);

    f = std::fopen((dir + "/lowcmd.csv").c_str(), "w");
    std::fprintf(f, "t_ns");
    for (int i = 0; i < kJoints; ++i) std::fprintf(f, ",q%d", i);
    std::fprintf(f, ",kp0,kd0,kp15\n");
    for (const auto &r : lc_rows)
    {
        std::fprintf(f, "%lld", (long long)r.t);
        for (float v : r.q) std::fprintf(f, ",%.9g", v);
        std::fprintf(f, ",%.9g,%.9g,%.9g\n", r.kp0, r.kd0, r.kp15);
    }
    std::fclose(f);

    f = std::fopen((dir + "/sport.csv").c_str(), "w");
    std::fprintf(f, "t_ns,x,y,z,vx,vy,vz\n");
    for (const auto &r : sp_rows)
        std::fprintf(f, "%lld,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", (long long)r.t, r.pos[0], r.pos[1], r.pos[2],
                     r.vel[0], r.vel[1], r.vel[2]);
    std::fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <out_dir> <seconds> [network=lo] [domain=0]\n", argv[0]);
        return 2;
    }
    const std::string out_dir = argv[1];
    const double seconds = std::stod(argv[2]);
    const std::string network = argc > 3 ? argv[3] : "lo";
    const int domain = argc > 4 ? std::stoi(argv[4]) : 0;

    std::signal(SIGINT, [](int) { stop_flag = true; });
    std::signal(SIGTERM, [](int) { stop_flag = true; });

    const size_t expected = size_t(seconds * 1100) + 1000;
    ls_rows.reserve(expected);
    lc_rows.reserve(expected);
    sp_rows.reserve(expected);

    unitree::robot::ChannelFactory::Instance()->Init(domain, network);
    unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_> lowstate("rt/lowstate");
    unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_> lowcmd("rt/lowcmd");
    unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_> sport("rt/sportmodestate");
    // queuelen 0: the handler runs in the DDS reader thread, so arrival times are not delayed by a queue.
    lowstate.InitChannel(on_lowstate);
    lowcmd.InitChannel(on_lowcmd);
    sport.InitChannel(on_sport);

    std::printf("probe: recording %.1f s on %s domain %d -> %s\n", seconds, network.c_str(), domain,
                out_dir.c_str());
    std::fflush(stdout);
    const int64_t end = now_ns() + int64_t(seconds * 1e9);
    while (!stop_flag && now_ns() < end) std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Write, then leave without closing the channels: CloseChannel / the DDS teardown crashed now and then
    // (segfault or abort, 22.10; once before anything was written, so the whole recording was lost). Handlers
    // wait on the mutexes meanwhile; the OS releases the DDS participant.
    std::scoped_lock lock(ls_mutex, lc_mutex, sp_mutex);
    write_csv(out_dir);
    std::printf("probe: lowstate %zu, lowcmd %zu, sport %zu rows\n", ls_rows.size(), lc_rows.size(),
                sp_rows.size());
    std::fflush(stdout);
    std::_Exit(0);
}
