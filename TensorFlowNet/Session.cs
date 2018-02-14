using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowNet
{
    public class Session
    {
        internal Session()
        {

        }

        public void Run(Operation op)
        {
            op.Execute();
        }

        public void Run(Operation op, Dictionary<string, Matrix<float>> feedDict)
        {
            
        }

        public Matrix<float>[] Run(params Tensor[] tensors)
        {
            Matrix<float>[] result = new Matrix<float>[tensors.Length];

            for (int i = 0; i < tensors.Length; i++)
            {
                result[i] = tensors[i].Evaluate(new Dictionary<string, Matrix<float>>());
            }

            return result;
        }

        public Matrix<float>[] Run(Tensor tensor)
        {
            return new Matrix<float>[] { tensor.Evaluate(new Dictionary<string, Matrix<float>>()) };
        }

        public Matrix<float>[] Run(Tensor tensor, Dictionary<string, Matrix<float>> feedDict)
        {
            return new Matrix<float>[] { tensor.Evaluate(feedDict) };
        }

        public Matrix<float>[] Run(Tensor[] tensors, Dictionary<string, Matrix<float>> feedDict)
        {
            Matrix<float>[] result = new Matrix<float>[tensors.Length];

            for (int i = 0; i < tensors.Length; i++)
            {
                result[i] = tensors[i].Evaluate(feedDict);
            }

            return result;
        }
    }
}
