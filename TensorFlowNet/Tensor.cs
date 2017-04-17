﻿using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowNet
{
    public abstract partial class Tensor
    {
        public string Identifier { get; internal set; }

        protected Type mDataType { get; set; }

        internal Tensor(Type dataType)
        {
            mDataType = dataType;
            Identifier = TensorName;
        }

        public abstract string TensorName { get; }

        // All evaluations and storages are done as float, but then on returning control to the client's program, do 
        // conversions based on the Tensor data types
        public abstract Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict);
        //public abstract object Evaluate(Dictionary<string, Matrix<float>> feedDict);

        public override string ToString()
        {
            return $"<Tensor \"{Identifier}\" datatype={mDataType.FullName}>";
        }
    }

    /// <summary>
    /// Constant tensors take no input, and output a vlue that it stores itnernally.
    /// </summary>
    /// <seealso cref="TensorFlowNet.Tensor" />
    public class ConstantTensor : Tensor
    {
        protected Matrix<float> mValue { get; set; }

        internal ConstantTensor(Matrix<float> value, Type dataType) : base(dataType)
        {
            mValue = value;
        }
        //protected object mValue { get; set; }

        //internal ConstantTensor(object value, Type dataType) : base(dataType)
        //{
        //    mValue = value;
        //}

        public override string TensorName => "Const";

        public override Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict)
        {
            return mValue;
        }

        public static implicit operator ConstantTensor(int value)
        {
            return new ConstantTensor(
                Matrix<float>.Build.DenseOfArray(new float[,] { { (float)value } }), typeof(int));
            //return new ConstantTensor(value, value.GetType());
        }

        public static implicit operator ConstantTensor(float value)
        {
            return new ConstantTensor(
                Matrix<float>.Build.DenseOfArray(new float[,] { { (float)value } }), typeof(float));
            //return new ConstantTensor(value, value.GetType());
        }
    }

    public class PlaceholderTensor : Tensor
    {
        public PlaceholderTensor(Type dataType) : base(dataType)
        {
        }

        public override string TensorName => "Placeholder";

        public override Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict)
        {
            //return feedDict[Identifier].Map(o => Convert.ChangeType(o, mDataType));
            return feedDict[Identifier];
        }
    }

    public class VariableTensor : Tensor
    {
        public readonly Matrix<float> InitialValue;

        public Matrix<float> Value { get; internal set; }

        public VariableTensor(Matrix<float> initialValue, Type dataType) : base(dataType)
        {
            InitialValue = initialValue;
        }

        public override string TensorName => "Variable";

        public override Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict)
        {
            return Value;
        }

        internal void Initialize()
        {
            Value = InitialValue;
        }
    }

    /// <summary>
    /// Addition Tensor outputs the sum of all the input tensors.
    /// </summary>
    /// <seealso cref="TensorFlowNet.Tensor" />
    public class AdditionTensor : Tensor
    {
        List<Tensor> mInputTensors { get; set; } = new List<Tensor>();
        public AdditionTensor(params Tensor[] inputs) : base(typeof(float))
        {
            mInputTensors = inputs.ToList();
        }

        public override string TensorName => "Add";

        public override Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict)
        {
            Matrix<float> sum;

            if (mInputTensors.Count > 0)
            {
                sum = mInputTensors.First().Evaluate(feedDict);
                foreach (Tensor inputTensor in mInputTensors.GetRange(1, mInputTensors.Count - 1))
                {
                    //sum += inputTensor.Evaluate(feedDict);
                    Matrix<float> evaluatedValue = inputTensor.Evaluate(feedDict);

                    // If it's a 1x1 matrix it's just a value for all intents and purposes
                    if (evaluatedValue.ColumnCount == 1 && evaluatedValue.RowCount == 1)
                    {
                        sum += evaluatedValue[0, 0];
                    }
                    else
                    {
                        sum += evaluatedValue;
                    }
                }
            }
            else
            {
                sum = Matrix<float>.Build.Dense(1, 1, 0);
            }
            return sum;
        }
    }

    /// <summary>
    /// Subraction Tensor ouputs the difference of all the input tensors.
    /// </summary>
    /// <seealso cref="TensorFlowNet.Tensor" />
    public class SubtractionTensor : Tensor
    {
        List<Tensor> mInputTensors { get; set; } = new List<Tensor>();
        public SubtractionTensor(params Tensor[] inputs) : base(typeof(float))
        {
            mInputTensors = inputs.ToList();
        }

        public override string TensorName => "Subtr";

        public override Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict)
        {
            Matrix<float> difference;

            if (mInputTensors.Count > 0)
            {
                difference = mInputTensors.First().Evaluate(feedDict);
                foreach (Tensor inputTensor in mInputTensors.GetRange(1, mInputTensors.Count - 1))
                {
                    //sum += inputTensor.Evaluate(feedDict);
                    Matrix<float> evaluatedValue = inputTensor.Evaluate(feedDict);

                    // If it's a 1x1 matrix it's just a value for all intents and purposes
                    if (evaluatedValue.ColumnCount == 1 && evaluatedValue.RowCount == 1)
                    {
                        difference -= evaluatedValue[0, 0];
                    }
                    else
                    {
                        difference -= evaluatedValue;
                    }
                }
            }
            else
            {
                difference = Matrix<float>.Build.Dense(1, 1, 0);
            }
            return difference;
        }
    }

    public class MultiplicationTensor : Tensor
    {
        List<Tensor> mInputTensors { get; set; } = new List<Tensor>();

        // Initialize to identity
        float mMultValue { get; set; } = 1;

        public MultiplicationTensor(params Tensor[] inputTensors) : base(typeof(float))
        {
            mInputTensors = inputTensors.ToList();
        }


        public MultiplicationTensor(float multValue, params Tensor[] inputTensors) : base(typeof(float))
        {
            mInputTensors = inputTensors.ToList();
            mMultValue = multValue;
        }

        public override string TensorName => "Mult";

        public override Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict)
        {
            Matrix<float> product;

            if (mInputTensors.Count > 0)
            {
                product = mInputTensors.First().Evaluate(feedDict);
                foreach (Tensor inputTensor in mInputTensors.GetRange(1, mInputTensors.Count - 1))
                {
                    product *= inputTensor.Evaluate(feedDict);
                }
                product *= mMultValue;
            }
            else
            {
                product = Matrix<float>.Build.Dense(1, 1, 1);
            }
            return product;
        }
    }

    public class SquareTensor : Tensor
    {
        Tensor mTensor { get; set; }
        public SquareTensor(Tensor tensor) : base(typeof(float))
        {
            mTensor = tensor;
        }

        public override string TensorName => "Square";

        public override Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict)
        {
            return mTensor.Evaluate(feedDict).Map(e => (float)Math.Pow(e, 2));
        }
    }

    public class DivisionTensor : Tensor
    {
        List<Tensor> mInputTensors { get; set; } = new List<Tensor>();

        // Initialize to identity
        float mDivValue { get; set; } = 1;


        public DivisionTensor(params Tensor[] inputTensors) : base(typeof(float))
        {
            mInputTensors = inputTensors.ToList();
        }

        public DivisionTensor(float divValue, params Tensor[] inputTensors) : base(typeof(float))
        {
            mInputTensors = inputTensors.ToList();
            mDivValue = divValue;
        }

        public override string TensorName => "Division";

        public override Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict)
        {
            Matrix<float> product;

            if (mInputTensors.Count > 0)
            {
                product = mInputTensors.First().Evaluate(feedDict);
                foreach (Tensor inputTensor in mInputTensors.GetRange(1, mInputTensors.Count - 1))
                {
                    product *= inputTensor.Evaluate(feedDict).Inverse();
                }
                product /= mDivValue;
            }
            else
            {
                product = Matrix<float>.Build.Dense(1, 1, 1);
            }
            return product;
        }
    }

    public class ReduceSumTensor : Tensor
    {
        public Tensor InputTensor { get; private set; }
        public ReduceSumAxis Axis { get; private set; }
        public bool KeepDimensions { get; private set; }
        public string Name { get; private set; }

        public ReduceSumTensor(Tensor inputTensor, ReduceSumAxis axis = ReduceSumAxis.None, bool keepDimensions = false, string name = "") : base(typeof(float))
        {
            InputTensor = inputTensor;
            Axis = axis;
            KeepDimensions = keepDimensions;
            Name = name;
        }

        public override string TensorName => "Sum";

        /// <summary>
        /// Evaluates the specified feed dictionary.
        /// </summary>
        /// <param name="feedDict">The feed dictionary.</param>
        /// <remarks>
        /// # 'x' is [[1, 1, 1]
        /// #         [1, 1, 1]]
        /// tf.reduce_sum(x) ==> 6
        /// tf.reduce_sum(x, 0) ==> [2, 2, 2]
        /// tf.reduce_sum(x, 1) ==> [3, 3]
        /// tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
        /// tf.reduce_sum(x, [0, 1]) ==> 6</remarks>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException">
        /// </exception>
        public override Matrix<float> Evaluate(Dictionary<string, Matrix<float>> feedDict)
        {
            Matrix<float> input = InputTensor.Evaluate(feedDict);
            Matrix<float> sum = Matrix<float>.Build.Dense(1, 1, 0);

            if (Axis == ReduceSumAxis.None)
            {
                for (int i = 0; i < input.RowCount; i++)
                {
                    for (int j = 0; j < input.ColumnCount; j++)
                    {
                        sum += input[i, j];
                    }
                }
            }
            else if (Axis == ReduceSumAxis.X)
            {
                if (!KeepDimensions)
                {
                    int endCount = input.RowCount;
                    sum = Matrix<float>.Build.Dense(1, endCount, 0);
                    for (int i = 0; i < endCount; i++)
                    {
                        float rowSum = 0f;
                        for (int j = 0; j < input.ColumnCount; j++)
                        {
                            rowSum += input[i, j];
                        }
                        sum[0, i] = rowSum;
                    }
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
            else if (Axis == ReduceSumAxis.Y)
            {
                if (!KeepDimensions)
                {
                    int endCount = input.ColumnCount;
                    sum = Matrix<float>.Build.Dense(1, endCount, 0);
                    for (int j = 0; j < endCount; j++)
                    {
                        float colSum = 0f;
                        for (int i = 0; i < input.RowCount; i++)
                        {
                            colSum += input[i, j];
                        }
                        sum[0, j] = colSum;
                    }
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
            else
            {
                throw new NotImplementedException();
            }

            return sum;
        }
    }

    public enum ReduceSumAxis
    {
        None,
        X,
        Y,
        Z
    }
}
