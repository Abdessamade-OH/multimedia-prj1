import { Component, OnInit, Input, ElementRef, ViewChild } from '@angular/core';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

@Component({
  selector: 'app-object-viewer',
  standalone: true,
  templateUrl: './object-viewer.component.html',
  styleUrl: './object-viewer.component.css'
})
export class ObjectViewerComponent implements OnInit {
  @Input() objectPath: string | undefined;
  @ViewChild('canvasContainer', { static: true }) canvasContainerRef!: ElementRef;

  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private objectGroup!: THREE.Group; // Group for the 3D object

  ngOnInit() {
    console.log("Received objectPath in ObjectViewerComponent:", this.objectPath);
    this.initScene();
  }

  ngAfterViewInit() {
    this.loadObject();
    this.animate();
  }

  private initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x222222);

    // ✅ Group to hold object
    this.objectGroup = new THREE.Group();
    this.scene.add(this.objectGroup);

    // ✅ Grid Helper
    const gridHelper = new THREE.GridHelper(10, 20, 0x888888, 0x444444);
    this.scene.add(gridHelper);

    // ✅ Set up camera
    const container = this.canvasContainerRef.nativeElement;
    this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    this.camera.position.set(0, 1, 5);

    // ✅ Set up renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    // ✅ Set up OrbitControls with object focus
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 50;
    this.controls.target.set(0, 0, 0); // Default target

    // ✅ Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.5);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
    directionalLight.position.set(5, 5, 5);
    this.scene.add(directionalLight);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.5);
    hemiLight.position.set(0, 10, 0);
    this.scene.add(hemiLight);
  }

  private loadObject() {
    if (!this.objectPath) {
      console.error("No objectPath provided!");
      return;
    }

    console.log("Loading 3D object from:", this.objectPath);

    const loader = new OBJLoader();
    loader.load(
      this.objectPath,
      (object) => {
        console.log("OBJ loaded successfully:", object);

        // ✅ Center the object
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        object.position.sub(center); // Move to center

        // ✅ Scale object properly
        const size = box.getSize(new THREE.Vector3()).length();
        const scaleFactor = 2 / size;
        object.scale.set(scaleFactor, scaleFactor, scaleFactor);

        // ✅ Assign materials if missing
        object.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            if (!Array.isArray(mesh.material)) {
              mesh.material = new THREE.MeshStandardMaterial({ color: 0xaaaaaa });
            }
          }
        });

        // ✅ Add object to group instead of scene
        this.objectGroup.clear(); // Remove previous object
        this.objectGroup.add(object);

        // ✅ Adjust camera to frame the object
        const boundingSize = box.getSize(new THREE.Vector3()).length();
        this.camera.position.set(0, boundingSize * 1.2, boundingSize * 2); // Adjust camera distance
        this.controls.target.copy(new THREE.Vector3(0, 0, 0)); // Orbit around object center
        this.controls.update();

        // ✅ Render scene
        this.renderer.render(this.scene, this.camera);
      },
      (xhr) => {
        console.log(`Loading progress: ${((xhr.loaded / xhr.total) * 100).toFixed(2)}%`);
      },
      (error) => {
        console.error("Error loading OBJ:", error);
      }
    );
  }

  private animate() {
    requestAnimationFrame(() => this.animate());

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
