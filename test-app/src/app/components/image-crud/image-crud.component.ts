import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';


@Component({
  selector: 'app-image-crud',
  standalone: true,
  imports: [ReactiveFormsModule, FormsModule, CommonModule],
  templateUrl: './image-crud.component.html',
  styleUrls: ['./image-crud.component.css']
})
export class ImageCrudComponent {

    imageCategory: string = 'aGrass'; // Default category
    selectedFiles: File[] = [];
    imagePreview: string | null = null;
    transformedImage: string | null = null;
    multipleFilesSelected: boolean = false;
    transforming: boolean = false; // Track if transformation section is visible
    rotation: number = 0;
    crop: number = 100; // 100% crop (full image)
    resize: number = 100; // 100% size (no resizing)
    grayscale: boolean = false;
    originalImage: string | null = null; // Store the original image for reset
  
    constructor(private imageService: ImageServiceService) {}
  
    onFileChange(event: any) {
      this.selectedFiles = event.target.files;
  
      const input = event.target as HTMLInputElement;
  
      if (input.files) {
        this.selectedFiles = Array.from(input.files);
  
        if (this.selectedFiles.length === 1) {
          const reader = new FileReader();
          reader.onload = () => {
            this.imagePreview = reader.result as string;
            this.originalImage = this.imagePreview; // Save original image
            this.transformedImage = this.imagePreview; // Set transformed image initially
          };
          reader.readAsDataURL(this.selectedFiles[0]);
          this.multipleFilesSelected = false;
        } else if (this.selectedFiles.length > 1) {
          this.imagePreview = null;
          this.multipleFilesSelected = true;
        }
      }
    }
  
    applyTransformations() {
      if (this.transformedImage) {
        let image = new Image();
        image.src = this.transformedImage;
  
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
  
        if (ctx) {
          const width = image.width * (this.resize / 100);
          const height = image.height * (this.resize / 100);
          canvas.width = width;
          canvas.height = height;
  
          // Reset the image and apply transformations in sequence
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(image, 0, 0, width, height);
  
          if (this.rotation) {
            ctx.save();
            ctx.translate(width / 2, height / 2);
            ctx.rotate((this.rotation * Math.PI) / 180);
            ctx.drawImage(image, -width / 2, -height / 2, width, height);
            ctx.restore();
          }
  
          if (this.grayscale) {
            ctx.filter = 'grayscale(100%)';
          }
  
          this.transformedImage = canvas.toDataURL();
        }
      }
    }
  
    // Function to reset transformations
    resetTransformations() {
      if (this.originalImage) {
        this.transformedImage = this.originalImage;
        this.rotation = 0;
        this.resize = 100;
        this.crop = 100;
        this.grayscale = false;
        this.applyTransformations();
      }
    }
  
    // Adjustments for rotation, resize, crop
    adjustRotation(value: number) {
      this.rotation += value;
      this.applyTransformations();
    }
  
    adjustResize(value: number) {
      this.resize += value;
      if (this.resize < 10) this.resize = 10; // Prevent resizing below 10%
      this.applyTransformations();
    }
  
    adjustCrop(value: number) {
      this.crop += value;
      if (this.crop < 10) this.crop = 10; // Prevent crop below 10%
      this.applyTransformations();
    }
  
    finishTransformation() {
      if (this.transformedImage) {
        const formData = new FormData();
        formData.append('category', this.imageCategory);
  
        // Ensure transformedImage is a valid string
        formData.append('images', this.transformedImage);
  
        this.imageService.uploadImage(formData).subscribe({
          next: (response) => {
            console.log('Transformed image uploaded successfully', response);
          },
          error: (error) => {
            console.error('Upload failed', error);
          }
        });
      } else {
        console.error('No transformed image available');
      }
    }
  
    cancelTransformations() {
      this.resetTransformations();
    }
  
    resetForm(): void {
      this.imageCategory = '';
      this.selectedFiles = [];
      this.imagePreview = null;
      this.transformedImage = null;
      this.multipleFilesSelected = false;
      this.transforming = false;
      this.rotation = 0;
      this.crop = 100;
      this.resize = 100;
      this.grayscale = false;
    }
  

  onUploadImage() {
    // Create FormData
    const formData = new FormData();

    // Append category first
    formData.append('category', this.imageCategory);

    // Append all selected files
    for (let i = 0; i < this.selectedFiles.length; i++) {
      formData.append('images', this.selectedFiles[i]);
    }

    // Call service method to upload
    this.imageService.uploadImage(formData).subscribe({
      next: (response) => {
        console.log('Upload successful', response);
        // Handle success
      },
      error: (error) => {
        console.error('Upload failed', error);
        // Handle error
      }
    });
  }

  onTransformImage() {
    // Show the transformation section
    this.transforming = true;
    this.transformedImage = this.imagePreview; // Start with the original image
  }

}
