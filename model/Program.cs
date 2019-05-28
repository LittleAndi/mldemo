using System;

namespace model
{
    class Program
    {
        static void Main(string[] args)
        {
            string dataPath = @"/Users/Lasse/dev/mldemo/model/data/iris.data";
            string modelPath = @"/Users/Lasse/dev/mldemo/model/model.zip";

            var model = Model.Train(new Microsoft.ML.LearningPipeline(), dataPath, modelPath).Result;

            var prediction = model.Predict(new IrisData
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f
            });

            System.Console.WriteLine($"Predicted flower type is {prediction.PredictedLabels}");
        }
    }
}
