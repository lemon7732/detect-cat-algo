"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from cat_rescue_ai.config import load_config
from cat_rescue_ai.detection.cat_face import CatFaceDetector
from cat_rescue_ai.gallery.index import build_gallery_index, load_gallery_index
from cat_rescue_ai.pipeline import RecognitionPipeline


def create_app(config_path: str | Path):
    from cat_rescue_ai.api.schemas import create_schema_namespace
    from cat_rescue_ai.exceptions import CatRescueAIError, DependencyNotAvailableError

    config = load_config(config_path)
    schemas = create_schema_namespace()
    app = FastAPI(
        title="Campus Stray Cat Recognition API",
        description="基于论文算法链路的校园流浪猫识别服务",
        version="0.1.0",
    )
    detector = CatFaceDetector(cascade_path=config["pipeline"].get("cascade_path"))
    pipeline: RecognitionPipeline | None = None

    def get_pipeline() -> RecognitionPipeline:
        nonlocal pipeline
        if pipeline is None:
            try:
                pipeline = RecognitionPipeline(
                    binary_config_path=config["pipeline"]["binary_config"],
                    landmarks_config_path=config["pipeline"]["landmarks_config"],
                    gallery_config_path=config["pipeline"]["gallery_config"],
                    cascade_path=config["pipeline"].get("cascade_path"),
                    allow_full_image_fallback=bool(config["pipeline"].get("allow_full_image_fallback", False)),
                )
            except DependencyNotAvailableError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"{exc}. Current endpoint requires TensorFlow model runtime. "
                        "Use Python 3.10-3.12 with requirements-ml.txt for full inference."
                    ),
                ) from exc
        return pipeline

    @app.get("/health", response_model=schemas["HealthResponse"], tags=["system"])
    async def health():
        return {"status": "ok"}

    @app.post(
        "/predict/species",
        response_model=schemas["SpeciesResponse"],
        tags=["prediction"],
        summary="判断图片是否为猫",
    )
    async def predict_species(file: UploadFile = File(...)):
        payload = await file.read()
        try:
            return get_pipeline().classify_species(payload)
        except CatRescueAIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/detect/cat-face",
        response_model=schemas["DetectionResponse"],
        tags=["prediction"],
        summary="检测图片中的猫脸",
    )
    async def detect_cat_face(file: UploadFile = File(...)):
        payload = await file.read()
        try:
            return detector.detect(payload)
        except CatRescueAIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/predict/landmarks",
        response_model=schemas["LandmarkResponse"],
        tags=["prediction"],
        summary="预测猫脸九点关键点",
    )
    async def predict_landmarks(file: UploadFile = File(...)):
        payload = await file.read()
        try:
            return get_pipeline().predict_landmarks(payload)
        except CatRescueAIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/identify/cat",
        response_model=schemas["IdentifyResponse"],
        tags=["prediction"],
        summary="进行 1:N 底库识别",
    )
    async def identify_cat(file: UploadFile = File(...)):
        payload = await file.read()
        try:
            return get_pipeline().identify(payload, top_k=int(config["matching"].get("top_k", 5)))
        except CatRescueAIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/gallery/rebuild",
        response_model=schemas["GalleryRebuildResponse"],
        tags=["gallery"],
        summary="重建校园猫底库索引",
    )
    async def rebuild_gallery():
        try:
            result = build_gallery_index(
                RecognitionPipeline(
                    binary_config_path=config["pipeline"]["binary_config"],
                    landmarks_config_path=config["pipeline"]["landmarks_config"],
                    cascade_path=config["pipeline"].get("cascade_path"),
                    allow_full_image_fallback=bool(config["pipeline"].get("allow_full_image_fallback", False)),
                ),
                config["pipeline"]["gallery_config"],
            )
            if pipeline is not None:
                pipeline.gallery_payload = load_gallery_index(config["pipeline"]["gallery_config"])
            return {"entries": len(result["entries"]), "failures": len(result["failures"])}
        except CatRescueAIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
