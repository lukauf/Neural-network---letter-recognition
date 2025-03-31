using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Lib
{
    internal class Neuron
    {
        protected List<Dendrite> _dendrites;
        double Bias { get; set; }

        public Neuron(int dentritesQtt)
        {
            _dendrites = new List<Dendrite>();
            for (int i = 0; i < dentritesQtt; i++)
            {
                _dendrites.Add(new Dendrite());
            }
        }

        public int Synapse()
        {
            double y_in = CalculateY_in();
            return CalculateOutput(y_in);
        }

        private double CalculateY_in ()
        {
            double y_in = 0;
            foreach (var dendrite in _dendrites)
            {
                if (dendrite.Value != null)
                {
                    y_in += (int)dendrite.Value * dendrite.Weight;
                }
            }
            return y_in;
        }

        private int CalculateOutput (double y_in)
        {
            return y_in > 0.5 ? 1 : -1;
        }
    }
}
