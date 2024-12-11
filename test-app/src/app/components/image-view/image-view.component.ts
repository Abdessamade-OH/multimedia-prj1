import { Component, OnInit } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-image-view',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './image-view.component.html',
  styleUrl: './image-view.component.css'
})
export class ImageViewComponent implements OnInit {
  images: any[] = []; // All images from the backend
  filteredImages: any[] = []; // Images filtered by category
  paginatedImages: any[] = []; // Images for the current page
  categories = ['aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'gParking'];

  currentPage = 1;
  itemsPerPage = 20;
  totalPages = 1;
  selectedCategories = new Set<string>();

  constructor(private imageService: ImageServiceService) {}

  ngOnInit(): void {
    this.fetchImages();
  }

  fetchImages(): void {
    this.imageService.getAllImages().subscribe({
      next: (data: any) => {
        this.images = data;
        this.filteredImages = [...this.images];
        this.calculatePagination();
      },
      error: (err) => {
        console.error('Error fetching images:', err);
      }
    });
  }

  onCategoryChange(event: Event): void {
    const checkbox = event.target as HTMLInputElement;
    if (checkbox.checked) {
      this.selectedCategories.add(checkbox.value);
    } else {
      this.selectedCategories.delete(checkbox.value);
    }

    this.applyFilters();
  }

  applyFilters(): void {
    if (this.selectedCategories.size > 0) {
      this.filteredImages = this.images.filter((img) =>
        this.selectedCategories.has(img.category)
      );
    } else {
      this.filteredImages = [...this.images];
    }

    this.calculatePagination();
  }

  calculatePagination(): void {
    this.totalPages = Math.ceil(this.filteredImages.length / this.itemsPerPage);
    this.changePage(1);
  }

  changePage(page: number): void {
    if (page < 1 || page > this.totalPages) return;
    this.currentPage = page;

    const startIndex = (page - 1) * this.itemsPerPage;
    const endIndex = startIndex + this.itemsPerPage;
    this.paginatedImages = this.filteredImages.slice(startIndex, endIndex);
  }

  get pagesToShow(): number[] {
    const totalPages = this.totalPages;
    const currentPage = this.currentPage;

    const pages = new Set<number>();
    pages.add(1); // First page
    pages.add(totalPages); // Last page

    for (let i = currentPage - 2; i <= currentPage + 2; i++) {
      if (i > 1 && i < totalPages) {
        pages.add(i);
      }
    }

    return Array.from(pages).sort((a, b) => a - b);
  }

  onImageError(event: Event): void {
    const img = event.target as HTMLImageElement;
    img.src = 'assets/default-image.jpg'; // Placeholder image
  }

}
