import { BrowserModule } from "@angular/platform-browser";
import { NgModule } from "@angular/core";
import { ImageCropperModule } from "ngx-image-cropper";
import { AppComponent } from "./app.component";
import { FormsModule } from "@angular/forms";
@NgModule({
  declarations: [AppComponent],
  imports: [BrowserModule, ImageCropperModule, FormsModule],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
