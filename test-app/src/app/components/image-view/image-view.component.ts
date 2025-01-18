import { Component, OnInit } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ObjectViewerComponent } from "../../shared/components/object-viewer/object-viewer.component";


@Component({
  selector: 'app-image-view',
  standalone: true,
  imports: [CommonModule, FormsModule, ObjectViewerComponent],
  templateUrl: './image-view.component.html',
  styleUrls: ['./image-view.component.css']
})
export class ImageViewComponent implements OnInit {

  constructor(private imageService: ImageServiceService) {}

  imageUrl: string | null = null;
  selectedCategory: string | null = 'Alabastron'; // Default category
  categoryImages: any[] = []; // To hold images fetched by category
  isModalOpen = false;
  modalImageName: string = '';
  modalObjectPath: string = '';  // To store the objectPath for the 3D viewer


  // 🔥 Fix: Categories array for dropdown options
  categories: string[] = [
    "Alabastron", "Amphora", "Amphoriskos", "Aryballos", "Askos", "Bowl",
    "Cup", "Dinos", "Epichysis", "Exaleiptron", "Skyphos", "Hydria", "Kalathos",
    "Kantharos", "Kernos", "Krater", "Kyathos", "Kylix", "Lagynos", "Lebes",
    "Lekane", "Lekythos", "Loutrophoros", "Lydion", "Mastos", "Mug", "Nestoris",
    "Oinochoe", "Pelike", "Pithos", "Plemochoe", "Psykter", "Pyxis", "Skyphos",
    "Other", "Modern-Bottle", "Modern-Vase", "Modern-Glass", "Modern-Bowl",
    "Modern-Cup", "Modern-Mug", "Modern-Urn", "Modern-Pot", "Pithoeidi",
    "Native American - Jar", "Native American - Effigy", "Native American - Bowl",
    "Native American - Bottle", "Picher Shaped", "Abstract"
  ];

  
  ngOnInit(): void {
    // Fetch images for the default category
    this.getImagesByCategory(this.selectedCategory!);
  }

  selectCategory(category: string): void {
    this.selectedCategory = category;
    this.getImagesByCategory(category);
  }

  getImagesByCategory(category: string): void {
    this.imageUrl = null;

    this.imageService.getImagesByCategory(category).subscribe({
      next: (images) => {
        this.categoryImages = images.map((image: { previewPath: string, objectPath: string }) => {
          const previewFilename = image.previewPath.split('\\').pop();
          const objectFilename = image.objectPath?.split('\\').pop(); // Extract filename

          return {
            ...image,
            previewPath: `http://localhost:3000/uploaded_images/${category}/previews/${previewFilename}`,
            objectPath: objectFilename
              ? `http://localhost:3000/uploaded_images/${category}/objects/${objectFilename}`
              : null
          };
        });

        console.log("Corrected image paths:", this.categoryImages);
      },
      error: (err) => {
        console.error('Error fetching images by category:', err);
        this.categoryImages = [];
      }
    });
  }

  deleteImage(id: string): void {
    this.imageService.deleteImageById(id).subscribe({
      next: () => {
        console.log('Image deleted successfully');
        
        // Refresh the category images
        if (this.selectedCategory) {
          this.getImagesByCategory(this.selectedCategory);
        }
      },
      error: (err) => {
        console.error('Error deleting image:', err);
      }
    });
  }
  
  
  getImageById(imageId: string): void {
    console.log('Fetching image with ID:', imageId); // Log the ID to verify it's correct
    this.imageService.getImageById(imageId).subscribe({
      next: (imageInfo) => {
        console.log('Image fetched:', imageInfo); // Log the fetched image data
        if (imageInfo && imageInfo.path) {
          const imagePath = imageInfo.path;
          const relativePath = imagePath.split('/src/upload_folder/')[1];
          const imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
          window.open(imageUrl, '_blank');
          console.log('Image opened:', imageUrl);
        } else {
          console.log('No image found');
        }
      },
      error: (err) => {
        console.error('Error fetching image by ID:', err);
      }
    });
  }
  

  openModal(image: any): void {
    this.modalImageName = image.name;
    this.modalObjectPath = image.objectPath;  // Pass the objectPath to the modal
    console.log(this.modalObjectPath);
    this.isModalOpen = true;
  }

  closeModal(): void {
    this.isModalOpen = false;
  }
  
}
