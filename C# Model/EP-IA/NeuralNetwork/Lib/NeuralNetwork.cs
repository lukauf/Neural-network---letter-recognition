using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkProject.Lib
{
    internal class NeuralNetwork
    {
        List<List<Neuron>> Layers;

        public NeuralNetwork()
        {
            int inputSize = Convert.ToInt32(ConfigurationManager.AppSettings["InputSize"]);
            int outputSize = Convert.ToInt32(ConfigurationManager.AppSettings["OutputSize"]);
            Layers = new List<List<Neuron>>();
            Layers.Add(GenerateEntryLayer(inputSize));
            Layers.Add(GenerateLayer(inputSize, 10));
            Layers.Add(GenerateLayer(10, outputSize));
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

        private List<Neuron> GenerateLayer(int dendriteQtt, int layerSize)
        {
            List<Neuron> layer = new List<Neuron>();
            for (int i = 0; i < layerSize; i++)
            {
                layer.Add(new Neuron(dendriteQtt));
            }
            return layer;
        }

        public List<char> Run(List<int> input)
        {
            List<char> output = new List<char>();
            List<int> layerOutput = new List<int>();
            List<int> previousLayerOutput;
            
            layerOutput = RunEntryLayer(input);

            for (int j = 1; j < Layers.Count; j++)
            {
                layerOutput = RunLayer(j, layerOutput);
            }

            previousLayerOutput = new List<int>(layerOutput);
            layerOutput.Clear();
            int i = 0;
            foreach (var value in previousLayerOutput)
            {
                foreach (var neuron in Layers[1])
                {
                    neuron.Dendrites[i].Value = value;
                }
                i++;
            }
            foreach (var neuron in Layers[1])
            {
                layerOutput.Add(neuron.Synapse());
            }

            previousLayerOutput = new List<int>(layerOutput);
            layerOutput.Clear();
            i = 0;
            foreach (var value in previousLayerOutput)
            {
                foreach (var neuron in Layers[2])
                {
                    neuron.Dendrites[i].Value = value;
                }
                i++;
            }
            foreach (var neuron in Layers[2])
            {
                layerOutput.Add(neuron.Synapse());
            }

            i = 0;
            foreach (var value in layerOutput)
            {
                if (value == 1)
                {
                    // converts output index to corresponding character
                    output.Add((char)(i + 65));
                }
                i++;
            }
            return output;
        }

        private List<int> RunEntryLayer(List<int> input)
        {
            List<int> layerOutput = new List<int>();
            int i = 0;
            foreach (var pixel in input)
            {
                Layers[0][i].Dendrites[0].Value = pixel;
                layerOutput.Add(Layers[0][i].Synapse());
                i++;
            }
            return layerOutput;
        }

        private List<int> RunLayer(List<int> previousLayerOutput)
        {
            List<int> layerOutput = new List<int>();
            int i = 0;
            foreach (var value in previousLayerOutput)
            {
                foreach (var neuron in Layers[1])
                {
                    neuron.Dendrites[i].Value = value;
                }
                i++;
            }
            foreach (var neuron in Layers[1])
            {
                layerOutput.Add(neuron.Synapse());
            }
            return layerOutput;
        }
    }
}
