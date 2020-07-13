import { Component } from "@angular/core";
import { ImageCroppedEvent } from "ngx-image-cropper";
import { FlaskRequestsService, ModelResponse } from "./flask-requests.service";

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
  picClassLogo: string;

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
      .subscribe(
        (res: ModelResponse) => {
          this.updatePrediction(res);
        },
        () => {
          console.error("API ERROR");
        }
      );
  }
  updatePrediction(prediction: ModelResponse) {
    let max = Object.keys(prediction.prediction).reduce((a, b) =>
      prediction.prediction[a] > prediction.prediction[b] ? a : b
    );
    console.log(prediction.prediction);
    this.picClass = max;
    this.picClassLogo = `../assets/logo_${max}.png`;
  }
}
