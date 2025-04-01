using NeuralNetworkProject.Lib;

namespace NeuralNetworkProject
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int[] bla = new[]
            {
                1, -1, 1, 1, 1, -1, -1, 1 -1, 1, 1, -1
            };
            List<int> test = new List<int>(bla);
            NeuralNetwork neuralNetwork = new NeuralNetwork();
            List<char> result = neuralNetwork.Run(test);
            result.ForEach(x => Console.WriteLine(x));
        }
    }
}
