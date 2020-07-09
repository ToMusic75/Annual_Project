import { Injectable } from "@angular/core";
import { HttpClient, HttpParams } from "@angular/common/http";
import { HttpHeaders } from "@angular/common/http";
import { Observable } from "rxjs";
import { catchError } from "rxjs/operators";

export interface ModelResponse {
  prediction: {
    renault: number;
    peugeot: number;
    volkswagen: number;
  };
}

export interface ModelPost {
  model_id: string;
  image: string;
}

@Injectable()
export class FlaskRequestsService {
  private httpOptions = {
    headers: new HttpHeaders({ "Content-type": "application/json" }),
  };
  private url_api = "http://localhost:5000";

  constructor(private http: HttpClient) {}

  sendPicture(post: ModelPost): Observable<ModelResponse> {
    console.log(post);
    return this.http.post<ModelResponse>(
      `${this.url_api}/predict`,
      post,
      this.httpOptions
    );
  }
}
