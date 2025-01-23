import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { catchError, map, Observable, of } from 'rxjs';
import { RemoveFirstLetterPipe } from "../../remove-first-letter.pipe";
import { Chart, ChartConfiguration, ChartData } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts';
import { ObjectViewerComponent } from "../../shared/components/object-viewer/object-viewer.component";
import { HttpClient } from '@angular/common/http';
import { forkJoin } from 'rxjs';


@Component({
  selector: 'app-simple-search',
  standalone: true,
  imports: [FormsModule, CommonModule, RemoveFirstLetterPipe, ObjectViewerComponent],
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

  isModalOpen = false;
  modalImageName: string = '';
  modalObjectPath: string = '';  // To store the objectPath for the 3D viewer
  image: any = null;


  constructor(private imageService: ImageServiceService, private http: HttpClient) {}

  
  openModal(image: any): void {
    this.modalImageName = image.name;
  
    if (!image.objectPath) {
      console.warn('No object path found for this image:', image);
      return;
    }
  
    // Prepend the correct base URL
    //this.modalObjectPath = `http://localhost:3000/${image.objectPath.replace(/\\/g, '/')}`;
  
    console.log(this.modalObjectPath);
    this.isModalOpen = true;
  }
  
  closeModal(): void {
    this.isModalOpen = false;
  }

  getImageByName(name: string): void {
    console.log('Starting image search...');
    this.isLoading = true;
  
    this.imageService.getImagesByName(name).subscribe({
      next: (imageInfo) => {
        console.log('Image search completed:', imageInfo); // Debugging log
  
        // Ensure imageInfo is an object and has the required properties
        if (imageInfo && imageInfo.previewPath) {
          let imagePath = imageInfo.previewPath;
  
          // Fix backslashes for URLs
          imagePath = imagePath.replace(/\\/g, '/');
  
          // Extract relative path after "upload_folder/"
          const relativePath = imagePath.split('upload_folder/')[1];
  
          // Construct the full image URL
          this.imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
  
          // Extract image name from the objectPath (last part of the path)
          const objectFileName = imageInfo.objectPath.split('\\').pop()?.split('/').pop();
          
          console.log('Extracted object file name:', objectFileName);
  
          // If we successfully extracted the file name, set it
          if (objectFileName) {
            this.imageName = objectFileName;
          } else {
            this.imageName = 'Unknown';
          }
  
          // Assign category
          this.imageCategory = imageInfo.category;
  
          console.log('Image URL:', this.imageUrl);
        } else {
          console.warn('No image found or imageInfo is missing previewPath.');
          this.imageUrl = null;
        }
  
        this.isLoading = false;
  
        this.image = {
          name: this.imageName,
          objectPath: imageInfo.objectPath // Ensure `imageInfo.objectPath` exists
        };
  
        // Construct the object path using name and category, appending .obj to the `imageInfo.name`
        this.modalObjectPath = `http://localhost:3000/uploaded_images/${imageInfo.category}/objects/${this.imageName}`;
  
        this.modalImageName = this.imageName;
      },
      error: (err) => {
        console.error('Error fetching image by name:', err);
        this.isLoading = false;
      },
    });
  }

  getImageByNameBase(name: string): Observable<string> { 
    console.log('Starting image search for: ', name);
  
    return this.imageService.getImagesByName(name).pipe(
      map((imageInfo) => {
        console.log('Image search completed:', imageInfo); // Debugging log
  
        // Ensure imageInfo is an object and has the required properties
        if (imageInfo && imageInfo.previewPath) {
          let imagePath = imageInfo.previewPath;
  
          // Fix backslashes for URLs
          imagePath = imagePath.replace(/\\/g, '/');
  
          // Extract relative path after "upload_folder/"
          const relativePath = imagePath.split('upload_folder/')[1];
  
          // Construct the full image URL
          const imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
          console.log('Constructed image URL:', imageUrl);
  
          return imageUrl;
        } else {
          console.error('Image info does not have previewPath');
          return ''; // Return an empty string if there's an issue
        }
      }),
      catchError((err) => {
        console.error('Error fetching image info:', err);
        return of(''); // Return an empty string in case of error
      })
    );
  }
  

  

  search_3d(objectPath: string, K: number = 5): void {
    console.log('Starting 3D search with objectPath:', objectPath, 'and K:', K);
    this.isLoading = true;
  
    console.log('Attempting to fetch file from URL:', objectPath);
    fetch(objectPath)
      .then((response) => {
        console.log('Received response from fetch:', response);
        if (!response.ok) {
          throw new Error(`Failed to fetch file: ${response.statusText}`);
        }
        return response.blob();
      })
      .then((blob) => {
        console.log('Fetched file as Blob:', blob);
  
        const file = new File([blob], 'model.obj', { type: blob.type });
        console.log('Created File object:', file);
  
        const formData = new FormData();
        formData.append('model', file, file.name);
        formData.append('n_results', K.toString());
  
        console.log('Sending file to backend API...');
  
        this.http.post<any>('http://localhost:5000/search_3d_model', formData).subscribe({
          next: (response: any) => {
            console.log('Search results received:', response);
  
            const imageRequests = response.results.map((result: any) => {
              console.log('wtf');
  
              const imagePath = result.thumbnail_path;
              console.log(imagePath);
  
              // Normalize backslashes to forward slashes
              const normalizedPath = imagePath.replace(/\\/g, '/');
  
              // Extract the last part of the path (file name)
              const fileNameWithExt = normalizedPath.split('/').pop() || '';
  
              // Ensure the file name has ".jpg"
              const finalName = fileNameWithExt.split('.')[0] + '.jpg';
  
              console.log("about to search image name", finalName);
  
              // Return an observable that fetches the image URL
              return this.getImageByNameBase(finalName).pipe(
                map((imageUrl) => ({
                  thumbnail_path: imageUrl,  // Store the correct image URL here
                  similarity: result.similarity,
                  category: result.category
                }))
              );
            });
  
            forkJoin<any[]>(imageRequests).subscribe({
              next: (updatedResults) => {
                this.similarImages = updatedResults;
                console.log('Formatted similar images:', this.similarImages);
              },
              error: (err) => {
                console.error('Error fetching image URLs:', err);
              },
              complete: () => {
                this.isLoading = false;
              }
            });
            
            
          },
          error: (err) => {
            console.error('Error during 3D search:', err);
            this.isLoading = false;
          }
        });
      })
      .catch((err) => {
        console.error('Error in fetch or processing file:', err);
        this.isLoading = false;
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
  const pathParts = imagePath.split('/'); // Split the URL by forward slashes
  const fileName = pathParts[pathParts.length - 1]; // Get the last part (the name with extension)
  const finalName= fileName.split('.')[0]; // Remove the extension by splitting by dot and taking the first part
  //console.log(finalName)
  return finalName
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
  
/*
  // Color distribution chart data
  colorChartData: ChartData<'bar'> = {
    labels: ['Red', 'Green', 'Blue'],
    datasets: [
      {
        data: this.features?.color_histogram ? [
          this.features.color_histogram.red.reduce((a: any, b: any) => a + b, 0), 
          this.features.color_histogram.green.reduce((a: any, b: any) => a + b, 0), 
          this.features.color_histogram.blue.reduce((a: any, b: any) => a + b, 0)
        ] : [0, 0, 0],
        backgroundColor: ['red', 'green', 'blue']
      }
    ]
  };

  // Texture GLCM Features chart data
  glcmChartData: ChartData<'bar'> = {
    labels: ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'],
    datasets: [
      {
        data: this.features?.glcm_features ? [
          this.features.glcm_features.contrast,
          this.features.glcm_features.dissimilarity,
          this.features.glcm_features.homogeneity,
          this.features.glcm_features.energy,
          this.features.glcm_features.correlation
        ] : [0, 0, 0, 0, 0],
        backgroundColor: '#007bff'
      }
    ]
  };

  // LBP Histogram chart data
  lbpChartData: ChartData<'bar'> = {
    labels: Array.from({ length: this.features?.lbp_features?.histogram.length }, (_, i) => `Bin ${i + 1}`),
    datasets: [
      {
        data: this.features?.lbp_features?.histogram || [],
        backgroundColor: '#28a745'
      }
    ]
  };

  ngAfterViewInit(): void {
    this.renderColorHistogramChart();
    this.renderDominantColorsChart();
    this.renderGLCMChart();
  }

  renderColorHistogramChart(): void {
    const ctx = document.getElementById('colorHistogramChart') as HTMLCanvasElement;
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Red', 'Green', 'Blue'],
        datasets: [
          {
            label: 'Color Distribution',
            data: [
              this.features.color_histogram.red.reduce((a: any, b: any) => a + b, 0),
              this.features.color_histogram.green.reduce((a: any, b: any) => a + b, 0),
              this.features.color_histogram.blue.reduce((a: any, b: any) => a + b, 0),
            ],
            backgroundColor: ['#ff0000', '#00ff00', '#0000ff'],
          },
        ],
      },
    });
  }

  renderDominantColorsChart(): void {
    const ctx = document.getElementById('dominantColorsChart') as HTMLCanvasElement;
    new Chart(ctx, {
      type: 'pie',
      data: {
        labels: this.features.dominant_colors.colors.map((_: any, index: number) => `Color ${index + 1}`),
        datasets: [
          {
            data: this.features.dominant_colors.percentages.map((p: number) => p * 100),
            backgroundColor: this.features.dominant_colors.colors.map(
              (rgb: any[]) => `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
            ),
          },
        ],
      },
    });
  }

  renderGLCMChart(): void {
    const ctx = document.getElementById('glcmChart') as HTMLCanvasElement;
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'],
        datasets: [
          {
            label: 'GLCM Features',
            data: [
              this.features.glcm_features.contrast.reduce((a: any, b: any) => a + b, 0),
              this.features.glcm_features.dissimilarity.reduce((a: any, b: any) => a + b, 0),
              this.features.glcm_features.homogeneity.reduce((a: any, b: any) => a + b, 0),
              this.features.glcm_features.energy.reduce((a: any, b: any) => a + b, 0),
              this.features.glcm_features.correlation.reduce((a: any, b: any) => a + b, 0),
            ],
            borderColor: '#007bff',
            fill: false,
          },
        ],
      },
    });
  }

  // Method to convert RGB arrays to string
  getRGBString(color: number[]): string {
    if (!color || color.length !== 3) return 'rgb(0, 0, 0)';
    return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
  }*/
  
}


