using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Lib
{
    internal class EntryNeuron : Neuron
    {
        public EntryNeuron(int dentritesQtt) : base(dentritesQtt)
        {
            _dendrites.Clear();
            _dendrites.Add(new Dendrite(0));
        }
    }
}
