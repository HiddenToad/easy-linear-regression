#![allow(non_upper_case_globals)]
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder, HttpResponseBuilder, http::StatusCode, http::header::ContentType};
use linear_regression::{LinearRegressionModel, TrainingResult};
use std::{fs::read_to_string, num::ParseFloatError};

fn do_train(req_body: String) -> Result<TrainingResult, ParseFloatError>{
    let mut model = LinearRegressionModel::new_uninit();

    let req_body = req_body
        .replace("data=", "")
        .replace(' ', "")
        .replace('(', "")
        .replace(')', "");
    
        for line in req_body.lines() {
            let line: Vec<&str> = line.trim().split(',').collect();
            model.add_points(&[(line[0].parse()?, line[1].parse()?)]);
        }

        model.set_epochs(model.graph.len() * 70000);

        model.train();

        Ok(model.best_result)
}

#[get("/")]
async fn index() -> impl Responder {
    HttpResponse::Ok().body(read_to_string("src/index.html").unwrap())
}

#[post("/train")]
async fn train(req_body: String) -> impl Responder {
    let result = do_train(req_body);

    match result{
        Ok(value) => {
            HttpResponseBuilder::new(StatusCode::OK)
                .json(value)
        }
        Err(_) => {
            HttpResponseBuilder::new(StatusCode::OK)
                .content_type(ContentType::plaintext())
                .body("INVALIDINPUT")
        }
    }

}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(index).service(train))
        .bind(("0.0.0.0", std::env::var("PORT").unwrap().parse().unwrap()))?
        .run()
        .await
}
