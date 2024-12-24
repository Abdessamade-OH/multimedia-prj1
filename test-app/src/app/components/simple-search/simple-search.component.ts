import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { catchError, map, Observable, of } from 'rxjs';
import { RemoveFirstLetterPipe } from "../../remove-first-letter.pipe";

@Component({
  selector: 'app-simple-search',
  standalone: true,
  imports: [FormsModule, CommonModule, RemoveFirstLetterPipe],
  templateUrl: './simple-search.component.html',
  styleUrls: ['./simple-search.component.css'],
})
export class SimpleSearchComponent {
  imageCategory!: string;
  numberK!: number;
  imageName: string = '';
  imageUrl: string | null = null;
  similarImages: any[] = [];
  isLoading: boolean = false; // Loading spinner state
  features: any = {}; // Store extracted features
  alpha!: number; // New alpha input
  beta!: number;   // New beta input
  gamma!: number;  // New gamma input
  imageSelections: any[] = []; // Array to store selected images (irrelevant)
  relevantSearch: boolean = false;

  relevanceSearchResults: any[] = []; // New array for relevance search results
  showRelevanceResults: boolean = false; // New flag to control relevance results visibility


  constructor(private imageService: ImageServiceService) {}

  getImageByName(name: string): void {
    console.log('Starting image search...');
    this.isLoading = true; // Start loading for image retrieval

    this.imageService.getImagesByName(name).subscribe({
      next: (imageInfo) => {
        console.log('Image search completed');
        if (imageInfo && imageInfo.length > 0) {
          const imagePath = imageInfo[0].path;
          const relativePath = imagePath.split('/src/upload_folder/')[1]; // Extract relative path
          this.imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
          const imageNameFromPath = relativePath.split('/').pop();
          this.imageName = imageNameFromPath || 'Unknown';
          this.imageCategory = imageInfo[0].category;
          

          this.isLoading = false; // Stop loading after image retrieval
        } else {
          console.log('No image found');
          this.imageUrl = null;
          this.isLoading = false; // Stop loading on no image found
        }
      },
      error: (err) => {
        console.error('Error fetching image by name:', err);
        this.isLoading = false; // Stop loading on error
      },
    });
  }

  getImageByName2(name: string): Observable<any> {
    console.log('Starting image search...');
    this.isLoading = true; // Start loading for image retrieval

    return this.imageService.getImagesByName(name).pipe(
      map((imageInfo) => {
        console.log('Image search completed');
        if (imageInfo && imageInfo.length > 0) {
          const imagePath = imageInfo[0].path;
          const relativePath = imagePath.split('/src/upload_folder/')[1]; // Extract relative path
          const imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
          const imageNameFromPath = relativePath.split('/').pop();
          const imageCategory = imageInfo[0].category;

          return { imageUrl, imageName: imageNameFromPath || 'Unknown', imageCategory };
        } else {
          console.log('No image found');
          return null; // Return null if no image found
        }
      }),
      catchError((err) => {
        console.error('Error fetching image by name:', err);
        return of(null); // Return null if there's an error
      })
    );
}

extractFeatures(): void {
  console.log('Starting feature extraction...');
  this.isLoading = true; // Start loading for feature extraction

  this.imageService.extractFeatures(this.imageName, this.imageCategory, this.numberK || 10).subscribe({
    next: (response) => {
      console.log(response);
      console.log('Feature extraction completed');
      this.similarImages = response.similar_images || []; // Extract similar images
      
      // Fetch the image for each similar image using getImageByName2
      let requestsCompleted = 0;
      const totalRequests = this.similarImages.length;

      this.similarImages.forEach((image: any) => {
        console.log(image);
        const imageName = this.extractImageName(image.image_path);
        this.getImageByName2(imageName).subscribe((imageData) => {
          if (imageData) {
            // Now you can use the imageData (imageUrl, imageName, imageCategory)
            console.log(imageData);
            // For example, you can update the image URL for each similar image
            image.url = imageData.imageUrl;
            image.name = imageData.imageName;
            this.features = response.features || {}; // Store extracted features
            console.log(this.features)
            image.category = imageData.imageCategory;
          }

          // Check if all requests have completed
          requestsCompleted++;
          if (requestsCompleted === totalRequests) {
            this.isLoading = false; // Stop loading after all requests are completed
          }
        });
      });

      this.relevantSearch = true;
      // If no similar images were found, stop loading immediately
      if (totalRequests === 0) {
        this.isLoading = false;
      }
    },
    error: (error) => {
      console.error('Error extracting features:', error);
      this.isLoading = false; // Stop loading on error
    },
  });
}

  extractImageName(imagePath: string): string {
    const pathParts = imagePath.split('\\'); // Split the path by backslashes
    return pathParts[pathParts.length - 1]; // Return the last part (the name)
  }

  toggleSelection(image: any): void {
    image.isSelected = !image.isSelected;
    if (image.isSelected) {
      // If image is selected, it is added to the irrelevant images list
      this.imageSelections.push(image);
    } else {
      // If image is deselected, remove from irrelevant images list
      this.imageSelections = this.imageSelections.filter((img) => img !== image);
    }
  }

  stripPrefix(imageName: string): string {
    const parts = imageName.split('-');
    return parts[parts.length - 1]; // Return the last part (the actual image name)
  }

  performRelevanceSearch(): void {
    console.log('Alpha:', this.alpha, 'Beta:', this.beta, 'Gamma:', this.gamma);
    this.isLoading = true;

    const relevantImages: string[] = [];
    const nonRelevantImages: string[] = [];
    let requestsCompleted = 0;
    const totalRequests = this.similarImages.length + 1;

    const constructImagePath = (imageData: any): string => {
      return `${imageData.imageCategory}/${imageData.imageName}`;
    };

    console.log(this.imageName);
    const mainImageName = this.extractImageName(this.imageName);
    console.log(mainImageName);
    
    this.getImageByName2(mainImageName).subscribe({
      next: (imageData) => {
        if (imageData) {
          relevantImages.push(constructImagePath(imageData));
          console.log('Main image data:', imageData);
          console.log('Main image path:', constructImagePath(imageData));
        }
        checkCompletion();
      },
      error: (err) => {
        console.error('Error fetching main image:', err);
        checkCompletion();
      },
    });

    this.similarImages
      .filter((img) => !this.imageSelections.includes(img))
      .forEach((img) => {
        const imageName = this.extractImageName(img.image_path);
        this.getImageByName2(imageName).subscribe({
          next: (imageData) => {
            if (imageData) {
              relevantImages.push(constructImagePath(imageData));
            }
            checkCompletion();
          },
          error: (err) => {
            console.error('Error fetching relevant image:', err);
            checkCompletion();
          },
        });
      });

    this.imageSelections.forEach((img) => {
      const imageName = this.extractImageName(img.image_path);
      this.getImageByName2(imageName).subscribe({
        next: (imageData) => {
          if (imageData) {
            nonRelevantImages.push(constructImagePath(imageData));
          }
          checkCompletion();
        },
        error: (err) => {
          console.error('Error fetching non-relevant image:', err);
          checkCompletion();
        },
      });
    });

    const checkCompletion = () => {
      requestsCompleted++;
      if (requestsCompleted === totalRequests) {
        const query = {
          name: this.imageName,
          category: this.imageCategory,
          relevant_images: relevantImages,
          non_relevant_images: nonRelevantImages,
          alpha: this.alpha,
          beta: this.beta,
          gamma: this.gamma
        };

        console.log('Query for relevance feedback:', query);

        this.imageService.sendRelevanceFeedback(query).subscribe({
          next: (response) => {
            console.log('Relevance feedback response:', response);
            // Store results in the new array instead of overwriting similarImages
            this.relevanceSearchResults = response.similar_images || [];
            
            let responseRequestsCompleted = 0;
            const totalResponseRequests = this.relevanceSearchResults.length;

            this.relevanceSearchResults.forEach((image: any) => {
              const imageName = this.extractImageName(image.image_path);
              this.getImageByName2(imageName).subscribe({
                next: (imageData) => {
                  if (imageData) {
                    image.url = imageData.imageUrl;
                    image.name = imageData.imageName;
                    image.category = imageData.imageCategory;
                  }
                  
                  responseRequestsCompleted++;
                  if (responseRequestsCompleted === totalResponseRequests) {
                    this.isLoading = false;
                    this.showRelevanceResults = true; // Show relevance results section
                  }
                },
                error: (err) => {
                  console.error('Error processing response image:', err);
                  responseRequestsCompleted++;
                  if (responseRequestsCompleted === totalResponseRequests) {
                    this.isLoading = false;
                    this.showRelevanceResults = true;
                  }
                }
              });
            });

            if (totalResponseRequests === 0) {
              this.isLoading = false;
              this.showRelevanceResults = true;
            }
          },
          error: (error) => {
            console.error('Error sending relevance feedback:', error);
            this.isLoading = false;
          },
        });
      }
    };
  }
  
  // Add this to your SimpleSearchComponent class
getRGBString(color: number[]): string {
  if (!color || color.length !== 3) return 'rgb(0, 0, 0)';
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}
}


