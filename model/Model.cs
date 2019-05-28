using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

class Model 
{
    public static async Task<PredictionModel<IrisData, IrisPrediction>> Train(LearningPipeline pipeline, string dataPath, string modelPath)
    {
        pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

        pipeline.Add(new Dictionarizer("Label"));

        pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

        pipeline.Add(new StochasticDualCoordinateAscentClassifier());

        pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel"});

        var model = pipeline.Train<IrisData, IrisPrediction>();

        await model.WriteAsync(modelPath);

        return model;
    }
}