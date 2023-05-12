from src.vodokanal.data.load_dataset import DataLoader
from src.vodokanal.data.transform_dataset import DataTransform
from src.vodokanal.models.create_model import CreateModel

if __name__ == '__main__':
    data = DataLoader()
    data_load = data.load_data()
    data_proccesed = DataTransform().initiate_data_transformation()

    obj = CreateModel()
    preprocessing_obj = CreateModel()._get_data_transformer_object()

    train_data, test_data = obj.preprocessing()
    model_score = obj.train_model(train_data, test_data)









    # data_transformation = DataTransformation()
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
    #     train_data, test_data
    # )
    #
    # model_trainer = ModelTrainer()
    # print(model_trainer.initiate_model_trainer(train_arr, test_arr))