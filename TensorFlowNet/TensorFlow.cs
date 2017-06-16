using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowNet
{
    public class TensorFlow
    {
        static Dictionary<string, Tensor> Tensors { get; set; } = new Dictionary<string, Tensor>();

        private static Tensor AddTensor(Tensor newTensor, int count)
        {
            newTensor.Identifier = $"{newTensor.TensorName}_{count}:0";
            Tensors.Add(newTensor.Identifier, newTensor);
            return newTensor;
        }

        public static ConstantTensor Constant<T>(T value)
        {
            return TensorFlow.Constant(value, typeof(T));
        }

        public static ConstantTensor Constant(object value, Type type)
        {
            float convertedValue = (float)Convert.ChangeType(value, typeof(float));
            ConstantTensor newTensor = new ConstantTensor(Matrix<float>.Build.Dense(1, 1, convertedValue), type);
            int count = Tensors.Count(t => t.Value is ConstantTensor);
            return (ConstantTensor)AddTensor(newTensor, count);
        }

        public static ConstantTensor Constant(Matrix<float> value)
        {
            ConstantTensor newTensor = new ConstantTensor(value, typeof(float));
            int count = Tensors.Count(t => t.Value is ConstantTensor);
            return (ConstantTensor)AddTensor(newTensor, count);
        }

        public static ConstantTensor Constant(params float[] values)
        {
            ConstantTensor newTensor = new ConstantTensor(Matrix<float>.Build.Dense(1, values.Length, (int x, int y) => values[y]), typeof(float));
            int count = Tensors.Count(t => t.Value is ConstantTensor);
            return (ConstantTensor)AddTensor(newTensor, count);
        }

        //public static AdditionTensor Add(params Tensor[] tensors)
        public static AdditionTensor Add(Tensor input1, Tensor input2)
        {
            //AdditionTensor newTensor = new AdditionTensor(tensors);
            AdditionTensor newTensor = new AdditionTensor(input1, input2);
            int count = Tensors.Count(t => t.Value is AdditionTensor);
            return (AdditionTensor)AddTensor(newTensor, count);
        }

        public static PlaceholderTensor Placeholder(Type type)
        {
            PlaceholderTensor newTensor = new PlaceholderTensor(type);
            int count = Tensors.Count(t => t.Value is PlaceholderTensor);
            return (PlaceholderTensor)AddTensor(newTensor, count);
        }

        public static VariableTensor Variable<T>(T initialValue)
        {
            float convertedValue = (float)Convert.ChangeType(initialValue, typeof(float));
            VariableTensor newTensor = new VariableTensor(Matrix<float>.Build.Dense(1, 1, convertedValue), typeof(T));
            int count = Tensors.Count(t => t.Value is VariableTensor);
            return (VariableTensor)AddTensor(newTensor, count);
        }

        public static ReduceSumTensor ReduceSum(Tensor tensor, ReduceSumAxis axis = ReduceSumAxis.None, bool keepDimensions = false, string name = "")
        {
            ReduceSumTensor newTensor = new ReduceSumTensor(tensor, axis, keepDimensions, name);
            int count = Tensors.Count(t => t.Value is ReduceSumTensor);
            return (ReduceSumTensor)AddTensor(newTensor, count);
        }

        public static VariableTensor Assign(VariableTensor tensor, float value)
        {
            Matrix<float> matrixValue = Matrix<float>.Build.Dense(1, 1, value);
            return TensorFlow.Assign(tensor, matrixValue);
        }

        public static VariableTensor Assign(VariableTensor tensor, Matrix<float> value)
        {
            VariableTensor t = (VariableTensor)Tensors[tensor.Identifier];
            t.Value = value;
            return t;
        }

        //public static MultiplicationTensor Square(Tensor tensor)
        //{
        //    return TensorFlow.Power(tensor, 2);
        //}

        //public static MultiplicationTensor Power(Tensor tensor, int power)
        //{
        //    return tensor.Pow(power);
        //}

        public static SquareTensor Square(Tensor tensor)
        {
            SquareTensor newTensor = new SquareTensor(tensor);
            int count = Tensors.Count(t => t.Value is SquareTensor);
            return (SquareTensor)AddTensor(newTensor, count);
        }

        public static VariableTensor Variable(Matrix<float> initialValue)
        {
            VariableTensor newTensor = new VariableTensor(initialValue, typeof(float));
            int count = Tensors.Count(t => t.Value is VariableTensor);
            return (VariableTensor)AddTensor(newTensor, count);
        }

        public static PowerTensor Pow(Tensor value, Tensor power)
        {
            PowerTensor newTensor = new PowerTensor(value, power);
            int count = Tensors.Count(t => t.Value is PowerTensor);
            return (PowerTensor)AddTensor(newTensor, count);
        }

        public static Session Session()
        {
            return new Session();
        }

        /// <summary>
        /// It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables. Until we call session.run, the variables are uninitialized.
        /// </summary>
        public static GlobalVariablesInitializerOperation GlobalVariablesInitializer()
        {
            return new GlobalVariablesInitializerOperation(Tensors.Where(x => x.Value is VariableTensor).Select(x => (VariableTensor)x.Value));
        }

        public static class Train
        {
            public static GradientDescentOptimizer GradientDescentOptimizer(float stepSize)
            {
                return new GradientDescentOptimizer(stepSize);
            }
        }

        public class GradientDescentOptimizer
        {
            public float StepSize { get; private set; }
            public GradientDescentOptimizer(float stepSize)
            {
                StepSize = stepSize;
            }

            public GradientDescentOperation Minimize(Tensor loss)
            {
                return new GradientDescentOperation(loss);
            }
        }
    }
}
