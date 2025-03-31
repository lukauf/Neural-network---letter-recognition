using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Lib
{
    internal class Dendrite
    {
        internal double Weight { get; set; }
        internal int? Value { get; set; }

        internal Dendrite()
        {
            Random random = new Random();
            Weight = random.NextDouble();
            Value = null;
        }

        internal Dendrite(double weight)
        {
            Weight = weight;
        }
    }
}
