using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkProject.Lib
{
    internal class Neuron
    {
        internal List<Dendrite> Dendrites;
        double Bias { get; set; }

        public Neuron(int dentritesQtt)
        {
            Dendrites = new List<Dendrite>();
            for (int i = 0; i < dentritesQtt; i++)
            {
                Dendrites.Add(new Dendrite());
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
            foreach (var dendrite in Dendrites)
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
            return y_in > 0.0 ? 1 : -1;
        }
    }
}
