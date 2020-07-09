import { BrowserModule } from "@angular/platform-browser";
import { NgModule } from "@angular/core";
import { ImageCropperModule } from "ngx-image-cropper";
import { AppComponent } from "./app.component";
import { FormsModule } from "@angular/forms";
import { FlaskRequestsService } from "./flask-requests.service";
import { HttpClientModule } from "@angular/common/http";
@NgModule({
  declarations: [AppComponent],
  imports: [BrowserModule, ImageCropperModule, FormsModule, HttpClientModule],
  providers: [FlaskRequestsService],
  bootstrap: [AppComponent],
})
export class AppModule {}
