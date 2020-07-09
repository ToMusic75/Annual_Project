import { Component } from "@angular/core";
import { ImageCroppedEvent } from "ngx-image-cropper";
import { FlaskRequestsService } from "./flask-requests.service";

@Component({
  selector: "app-root",
  templateUrl: "./app.component.html",
  styleUrls: ["./app.component.scss"],
})
export class AppComponent {
  title = "frontend";
  modelOptions: string[] = [
    "linear_model",
    "mlp_model",
    "svm_model",
    "cnn_model",
  ];
  imageChangedEvent: any = "";
  croppedImage: any = "";
  isImage: boolean = false;
  selectedModel: string = "mlp_model";
  picClass: string;

  constructor(private flaskRequestsService: FlaskRequestsService) {}

  fileChangeEvent(event: any): void {
    this.imageChangedEvent = event;
  }

  imageCropped(event: ImageCroppedEvent) {
    this.croppedImage = event.base64;
    this.isImage = true;
  }

  loadImageFailed() {
    window.alert("Fail to load image");
  }
  submitImage() {
    this.flaskRequestsService
      .sendPicture({
        model_id: this.selectedModel,
        image: this.croppedImage,
      })
      .subscribe((res) => {
        console.log(res);
      });
  }
}
