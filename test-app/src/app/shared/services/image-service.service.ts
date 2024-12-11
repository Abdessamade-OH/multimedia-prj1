import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ImageServiceService {

  constructor(private http: HttpClient) { }

  private apiUrl = 'http://localhost:3000/api/images'; // Your backend API URL

  // Upload image method using FormData
  uploadImage(formData: FormData): Observable<any> {
    return this.http.post(`${this.apiUrl}/upload`, formData);
  }

  // Delete image by name
  deleteImage(imageName: string): Observable<any> {
    return this.http.delete(`${this.apiUrl}/delete/${imageName}`);
  }


  // Simple search by image name
  simpleSearch(imageName: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/search/simple`, { params: { name: imageName } });
  }

  // Advanced search with additional criteria
  advancedSearch(imageName: string, category: string, otherFilters: any): Observable<any> {
    return this.http.get(`${this.apiUrl}/search/advanced`, {
      params: { 
        name: imageName,
        category: category,
        ...otherFilters
      }
    });
  }

  getAllImages() {
    return this.http.get(`${this.apiUrl}/all`); // Adjust endpoint based on your backend
  }
}
