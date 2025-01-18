import { Component, OnInit, Input, ElementRef, ViewChild } from '@angular/core';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; // ✅ Import OrbitControls

@Component({
  selector: 'app-object-viewer',
  standalone: true,
  imports: [],
  templateUrl: './object-viewer.component.html',
  styleUrl: './object-viewer.component.css'
})
export class ObjectViewerComponent implements OnInit {
  @Input() objectPath: string | undefined; // Path to the 3D object
  @ViewChild('canvasContainer', { static: true }) canvasContainerRef!: ElementRef;

  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;

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
    this.scene.background = new THREE.Color(0x222222); // ✅ Dark gray background instead of pure black

    // ✅ Add a Grid Helper
    const gridHelper = new THREE.GridHelper(10, 20, 0x888888, 0x444444); // Grid size 10, 20 divisions
    this.scene.add(gridHelper);

    // ✅ Set up camera
    const container = this.canvasContainerRef.nativeElement;
    this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    this.camera.position.set(0, 1, 5); // Initial position

    // ✅ Set up renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    // ✅ Add OrbitControls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 50;

    // ✅ Improved lighting
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
        object.position.sub(center);

        // ✅ Scale object if too big or small
        const size = box.getSize(new THREE.Vector3()).length();
        const scaleFactor = 2 / size;
        object.scale.set(scaleFactor, scaleFactor, scaleFactor);

        // ✅ Ensure object has visible material
        object.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            if (!Array.isArray(mesh.material)) {
              mesh.material = new THREE.MeshStandardMaterial({ color: 0xaaaaaa });
            }
          }
        });

        // ✅ Add the object to the scene
        this.scene.add(object);

        // ✅ Adjust camera to frame the object
        const boundingSize = box.getSize(new THREE.Vector3()).length();
        this.camera.position.set(0, boundingSize * 1.2, boundingSize * 2); // Adjust camera distance
        this.controls.target.copy(center); // Focus on object
        this.controls.update();

        // Render scene
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
