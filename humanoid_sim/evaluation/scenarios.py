"""Versioned, code-owned protocols; search agents cannot rewrite final tests."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    schedule: tuple

    def command_at(self,seconds):
        command=self.schedule[0][1]
        for start,value in self.schedule:
            if seconds<start:break
            command=value
        return command


@dataclass(frozen=True)
class Suite:
    name: str
    seeds: tuple[int,...]
    seconds: float
    scenarios: tuple[str,...]
    version: str='g1-command-v2'


SCENARIOS={name:Scenario(name,((0.,(speed,0.,0.)),)) for name,speed in
           [('stand',0.),('slow',.5),('walk',1.),('brisk',1.4)]}
SCENARIOS['stop_start']=Scenario('stop_start',((0.,(0.,0.,0.)),(5.,(1.,0.,0.)),
                                             (10.,(0.,0.,0.)),(15.,(1.,0.,0.))))
# Sprint branches observe a constant 2 m/s command; not part of the versioned command suites.
SCENARIOS['sprint']=Scenario('sprint',((0.,(2.,0.,0.)),))
SUITE_SCENARIOS=('stand','slow','walk','brisk','stop_start')


def get_scenario(name):
    if name not in SCENARIOS:raise ValueError(f'Unknown scenario: {name}')
    return SCENARIOS[name]


def get_suite(name):
    settings={'screen':(90000,5,20.),'validation':(100000,20,60.),'final':(200000,30,60.)}
    if name not in settings:raise ValueError(f'Unknown evaluation split: {name}')
    start,count,seconds=settings[name]
    return Suite(name,tuple(range(start,start+count)),seconds,SUITE_SCENARIOS)
