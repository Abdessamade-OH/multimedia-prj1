import { Injectable } from '@angular/core';
import { Image as ImageJs } from 'image-js';

@Injectable({
  providedIn: 'root'
})
export class ImageTransformService {

  constructor() {}

  // Rotate the image by 90 degrees left or right
  async rotateImage(imageBase64: string, direction: 'left' | 'right') {
    const image = await ImageJs.load(imageBase64);

    const rotatedImage = direction === 'left' ? image.rotate(-90) : image.rotate(90);

    return rotatedImage.toDataURL();
  }

  // Resize the image, maintaining aspect ratio or forcing specific width
  async resizeImage(imageBase64: string, width: number, height?: number) {
    const image = await ImageJs.load(imageBase64);

    const resizedImage = height
      ? image.resize({ width, height })  // Custom width and height
      : image.resize({ width });          // Only custom width, height adjusted to maintain aspect ratio

    return resizedImage.toDataURL();
  }

  // Crop the image (x, y, width, height)
  async cropImage(imageBase64: string, x: number, y: number, width: number, height: number) {
    const image = await ImageJs.load(imageBase64);

    const croppedImage = image.crop({ x, y, width, height });

    return croppedImage.toDataURL();
  }

  // Convert the image to grayscale
  async grayscaleImage(imageBase64: string) {
    const image = await ImageJs.load(imageBase64);

    const grayscaleImage = image.grey();

    return grayscaleImage.toDataURL();
  }
}
