import { Component } from "@angular/core";
import { ImageCroppedEvent } from "ngx-image-cropper";

@Component({
  selector: "app-root",
  templateUrl: "./app.component.html",
  styleUrls: ["./app.component.scss"],
})
export class AppComponent {
  title = "frontend";
  modelOptions: string[] = ["1", "2", "3"];
  imageChangedEvent: any = "";
  croppedImage: any = "";
  isImage: boolean = false;
  selectedModel: string = "1";
  picClass: string;

  fileChangeEvent(event: any): void {
    this.imageChangedEvent = event;
  }

  imageCropped(event: ImageCroppedEvent) {
    this.croppedImage = event.base64;
    this.isImage = true;
  }

  imageLoaded() {
    // show cropper
  }

  loadImageFailed() {
    window.alert("Fail to load image");
  }
  submitImage() {
    console.log("cc");
  }
}
