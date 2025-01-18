import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ImageTransformService } from '../../shared/services/image-transform.service';

@Component({
  selector: 'app-image-crud',
  standalone: true,
  imports: [ReactiveFormsModule, FormsModule, CommonModule],
  templateUrl: './image-crud.component.html',
  styleUrls: ['./image-crud.component.css']
})
export class ImageCrudComponent {

  imageCategory: string = 'Alabastron'; // Default category
  selectedImageFiles: File[] = [];
  selectedObjFiles: File[] = [];
  imagePreview: string | null = null;
  multipleFilesSelected: boolean = false;
  objFilesMatch: boolean = true; // Track if image files and obj files match

  // Long list of categories
  categories = [
    'Alabastron', 'Amphora', 'Amphoriskos', 'Aryballos', 'Askos', 'Bowl', 'Cup', 'Dinos', 'Epichysis',
    'Exaleiptron', 'Skyphos', 'Hydria', 'Kalathos', 'Kantharos', 'Kernos', 'Krater', 'Kyathos', 'Kylix',
    'Lagynos', 'Lebes', 'Lekane', 'Lekythos', 'Loutrophoros', 'Lydion', 'Mastos', 'Mug', 'Nestoris',
    'Oinochoe', 'Pelike', 'Pithos', 'Plemochoe', 'Psykter', 'Pyxis', 'Skyphos', 'Other', 'Modern-Bottle',
    'Modern-Vase', 'Modern-Glass', 'Modern-Bowl', 'Modern-Cup', 'Modern-Mug', 'Modern-Urn', 'Modern-Pot',
    'Pithoeidi', 'Native American - Jar', 'Native American - Effigy', 'Native American - Bowl',
    'Native American - Bottle', 'Picher Shaped', 'Abstract'
  ];

  constructor(private imageService: ImageServiceService, private imageTransformService: ImageTransformService) {}

  onFileChange(event: any, fileType: 'image' | 'obj') {
    if (fileType === 'image') {
      this.selectedImageFiles = event.target.files;
      this.imagePreview = null;
      if (this.selectedImageFiles.length === 1) {
        const reader = new FileReader();
        reader.onload = () => {
          this.imagePreview = reader.result as string;
        };
        reader.readAsDataURL(this.selectedImageFiles[0]);
      }
    } else if (fileType === 'obj') {
      this.selectedObjFiles = event.target.files;
      if (this.selectedImageFiles.length !== this.selectedObjFiles.length) {
        this.objFilesMatch = false;
      } else {
        this.objFilesMatch = true;
      }
    }
  }

  onUploadImage() {
    if (this.selectedImageFiles.length !== this.selectedObjFiles.length) {
      console.error('Number of images and OBJ files do not match!');
      return;
    }

    const formData = new FormData();
    formData.append('category', this.imageCategory);

    // Append image files under the "previews" field name
    for (let i = 0; i < this.selectedImageFiles.length; i++) {
      formData.append('previews', this.selectedImageFiles[i]);
    }

    // Append OBJ files under the "objects" field name
    for (let i = 0; i < this.selectedObjFiles.length; i++) {
      formData.append('objects', this.selectedObjFiles[i]);
    }

    // Call service method to upload
    this.imageService.uploadImage(formData).subscribe({
      next: (response) => {
        console.log('Upload successful', response);
      },
      error: (error) => {
        console.error('Upload failed', error);
      }
    });
  }

  resetForm(): void {
    this.imageCategory = 'Alabastron'; // Reset to default category
    this.selectedImageFiles = [];
    this.selectedObjFiles = [];
    this.imagePreview = null;
    this.multipleFilesSelected = false;
    this.objFilesMatch = true; // Reset match flag
  }
}
