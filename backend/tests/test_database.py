from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.db.models import Base, Dataset, Model, Detection, Explanation

def test_database_models_and_relationships():
    engine=create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        ds=Dataset(id="nsl-kdd", name="NSL-KDD")
        db.add(ds); db.flush()
        model=Model(dataset_id=ds.id, name="random_forest", task="binary", artifact_key="nsl-kdd/binary/random_forest")
        db.add(model); db.flush()
        det=Detection(dataset_id=ds.id, model_id=model.id, prediction="normal", prediction_index=0, probability=.9, probabilities={"normal":.9,"attack":.1}, raw_features={"x":1})
        db.add(det); db.flush()
        db.add(Explanation(detection_id=det.id, method="shap", payload={"features":[]}))
        db.commit()
        assert db.get(Detection, det.id).model.name == "random_forest"
        assert db.get(Detection, det.id).explanations[0].method == "shap"
