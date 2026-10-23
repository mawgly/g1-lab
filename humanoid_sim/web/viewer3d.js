// Shared WebGL scene for MuJoCo geometry: lights, floor grid, robot meshes and per-frame transforms.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { t } from './i18n.js';

export { THREE };

// Creates renderer, camera, controls and lights inside `viewport`; `onFrame(sun)` runs before each render.
export function createViewer(viewport, onFrame = () => {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  viewport.prepend(renderer.domElement);
  renderer.domElement.setAttribute('aria-label', t('Трёхмерная модель робота, управляемая физикой MuJoCo'));
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog('#1c272c', 10, 34);
  // A near plane of 5 cm (not 2 cm) keeps enough depth precision for the grid lines just above the floor.
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 100);
  camera.up.set(0, 0, 1);
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.08;
  orbit.minDistance = 0.25;
  orbit.maxDistance = 14;
  orbit.maxPolarAngle = Math.PI / 2 - 0.025;
  scene.add(new THREE.HemisphereLight('#d7f2e6', '#303a3e', 2.5));
  const sun = new THREE.DirectionalLight('#f2f1d8', 4);
  sun.position.set(3, -4, 8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -5;
  sun.shadow.camera.right = sun.shadow.camera.top = 5;
  sun.shadow.normalBias = 0.02;
  sun.shadow.bias = -0.0002;
  sun.shadow.radius = 4;
  scene.add(sun, sun.target);
  const rim = new THREE.DirectionalLight('#a5c3e6', 2.2);
  rim.position.set(-3, 2, 4);
  scene.add(rim);
  // Floor grid: 1 m cells with stronger lines every 5 m, so walking speed can be judged by eye.
  // Lines must be lighter than the lit floor; the MuJoCo checker texture is not rendered here.
  // The grid and the floor (see addGeometry) move with the camera target in 5 m steps, so lines stay on
  // whole metres and a runner never leaves them. The floor stays a moderate plane: a kilometres-wide
  // one lost depth precision and the whole grid flickered under it.
  const grids = [];
  for (const [divisions, color, opacity, z] of [
    [200, '#9fc0bd', 0.28, 0.005],
    [40, '#d7ece6', 0.55, 0.007],
  ]) {
    const grid = new THREE.GridHelper(200, divisions, color, color);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = z;
    grid.material.transparent = true;
    grid.material.opacity = opacity;
    grid.material.depthWrite = false;
    scene.add(grid);
    grids.push(grid);
  }
  scene.userData.followers = grids;
  new ResizeObserver(() => {
    const width = viewport.clientWidth,
      height = viewport.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }).observe(viewport);
  renderer.setAnimationLoop(() => {
    orbit.update();
    onFrame(sun);
    scene.userData.followers = scene.userData.followers.filter((object) => object.parent); // drop removed floors
    for (const object of scene.userData.followers) {
      object.position.x = Math.round(orbit.target.x / 5) * 5;
      object.position.y = Math.round(orbit.target.y / 5) * 5;
    }
    renderer.render(scene, camera);
  });
  return { renderer, scene, camera, orbit, sun };
}

function robotGeometry(geom) {
  let geometry;
  if (geom.type === 2) geometry = new THREE.SphereGeometry(geom.size[0], 24, 16);
  else if (geom.type === 3) {
    geometry = new THREE.CapsuleGeometry(geom.size[0], 2 * geom.size[1], 8, 20);
    geometry.rotateX(Math.PI / 2); // MuJoCo capsules extend along local Z.
  } else if (geom.type === 4) {
    geometry = new THREE.SphereGeometry(1, 24, 16);
    geometry.scale(...geom.size);
  } else if (geom.type === 5) {
    geometry = new THREE.CylinderGeometry(geom.size[0], geom.size[0], 2 * geom.size[1], 24);
    geometry.rotateX(Math.PI / 2);
  } else if (geom.type === 6) geometry = new THREE.BoxGeometry(...geom.size.map((x) => 2 * x));
  else if (geom.type === 7) {
    geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(geom.vertices, 3));
    geometry.setIndex(geom.faces);
    geometry.computeVertexNormals();
  } else throw new Error(t('Неизвестный тип геометрии {type}', { type: geom.type }));
  return geometry;
}

// Adds meshes for /api/model geometry; returns them indexed by geom id. `tint` blends robot colours
// towards a colour (race lanes); `floor: false` skips plane geoms when several robots share a floor;
// a shared `cache` Map reuses vertex buffers between robots of the same model.
export function addGeometry(scene, model, { floor = true, tint = null, cache = null } = {}) {
  const meshes = [];
  for (const geom of model) {
    let geometry, material;
    if (geom.type === 0) {
      if (!floor) continue;
      geometry = new THREE.PlaneGeometry(400, 400); // follows the camera like the grid
      material = new THREE.MeshStandardMaterial({ color: '#253238', roughness: 0.97, metalness: 0 });
    } else {
      geometry = cache?.get(geom.id) ?? robotGeometry(geom);
      cache?.set(geom.id, geometry);
      const color = new THREE.Color(...geom.color.slice(0, 3));
      if (tint) color.lerp(new THREE.Color(tint), 0.45);
      material = new THREE.MeshStandardMaterial({
        color,
        opacity: geom.color[3],
        transparent: geom.color[3] < 1,
        roughness: 0.48,
        metalness: 0.15,
      });
    }
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = geom.type !== 0;
    mesh.receiveShadow = true;
    scene.add(mesh);
    if (geom.type === 0) {
      mesh.userData.floor = true; // placed by the viewer, not by MuJoCo transforms
      scene.userData.followers?.push(mesh);
    }
    meshes[geom.id] = mesh;
  }
  return meshes;
}

// Applies [x, y, z, qx, qy, qz, qw] transforms; offsetY shifts a robot sideways onto its lane.
export function applyGeoms(meshes, geoms, offsetY = 0) {
  for (let i = 0; i < meshes.length; i++) {
    const transform = geoms[i];
    if (!transform || !meshes[i] || meshes[i].userData.floor) continue;
    meshes[i].position.fromArray(transform);
    meshes[i].position.y += offsetY;
    meshes[i].quaternion.fromArray(transform, 3);
  }
}
