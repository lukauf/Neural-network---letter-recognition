using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Lib
{
    internal class NeuralNetwork
    {
        List<List<Neuron>> Layers;

        public NeuralNetwork()
        {
            int inputSize = Convert.ToInt32(ConfigurationManager.AppSettings["InputSize"]);
            Layers = new List<List<Neuron>>();
            Layers.Add(GenerateEntryLayer(inputSize));
        }

        private List<Neuron> GenerateEntryLayer(int layerSize)
        {
            List<Neuron> entryLayer = new List<Neuron>();
            for (int i = 0; i < layerSize; i++)
            {
                entryLayer.Add(new EntryNeuron(layerSize));
            }
            return entryLayer;
        }

    }
}
