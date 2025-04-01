using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkProject.Lib
{
    internal class EntryNeuron : Neuron
    {
        public EntryNeuron(int dentritesQtt) : base(dentritesQtt)
        {
            Dendrites.Clear();
            Dendrites.Add(new Dendrite(1));
        }
    }
}
