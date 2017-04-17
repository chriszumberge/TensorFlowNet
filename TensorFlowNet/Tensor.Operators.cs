using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowNet
{
    public abstract partial class Tensor
    {
        public static AdditionTensor operator +(Tensor a, Tensor b)
        {
            return new AdditionTensor(a, b);
        }

        public static AdditionTensor operator +(Tensor a, int value)
        {
            return new AdditionTensor(a, (ConstantTensor)value);
        }

        public static AdditionTensor operator +(int value, Tensor a)
        {
            return new AdditionTensor((ConstantTensor)value, a);
        }

        public static AdditionTensor operator +(Tensor a, float value)
        {
            return new AdditionTensor(a, (ConstantTensor)value);
        }

        public static AdditionTensor operator +(float value, Tensor a)
        {
            return new AdditionTensor((ConstantTensor)value, a);
        }

        public static SubtractionTensor operator -(Tensor a, Tensor b)
        {
            return new SubtractionTensor(a, b);
        }

        public static SubtractionTensor operator -(Tensor a, int value)
        {
            return new SubtractionTensor(a, (ConstantTensor)value);
        }

        public static SubtractionTensor operator -(int value, Tensor a)
        {
            return new SubtractionTensor((ConstantTensor)value, a);
        }

        public static SubtractionTensor operator -(Tensor a, float value)
        {
            return new SubtractionTensor(a, (ConstantTensor)value);
        }

        public static SubtractionTensor operator -(float value, Tensor a)
        {
            return new SubtractionTensor((ConstantTensor)value, a);
        }

        public static MultiplicationTensor operator *(Tensor a, Tensor b)
        {
            return new MultiplicationTensor(a, b);
        }

        public static MultiplicationTensor operator *(int multiplicity, Tensor a)
        {
            return new MultiplicationTensor(multiplicity, a);
        }

        public static MultiplicationTensor operator *(Tensor a, int multiplicity)
        {
            return new MultiplicationTensor(multiplicity, a);
        }

        public static MultiplicationTensor operator *(float multiplicity, Tensor a)
        {
            return new MultiplicationTensor(multiplicity, a);
        }

        public static MultiplicationTensor operator *(Tensor a, float multiplicity)
        {
            return new MultiplicationTensor(multiplicity, a);
        }

        //public static DivisionTensor operator /(Tensor a, Tensor b)
        //{
        //    return new DivisionTensor(a, b);
        //}

        //public static DivisionTensor operator /(int diviser, Tensor a)
        //{
        //    return new DivisionTensor(diviser, a);
        //}

        //public static DivisionTensor operator /(Tensor a, int diviser)
        //{
        //    return new DivisionTensor(a, (ConstantTensor)diviser);
        //}

        //public static DivisionTensor operator /(float diviser, Tensor a)
        //{
        //    return new DivisionTensor(diviser, a);
        //}

        //public static DivisionTensor operator /(Tensor a, float diviser)
        //{
        //    return new DivisionTensor(a, (ConstantTensor)diviser);
        //}
    }

    public static class TensorExtensions
    {
        //public static MultiplicationTensor Pow(this Tensor a, int exponential)
        //{
        //    Tensor[] tensors = new Tensor[exponential];
        //    for (int i = 0; i < exponential; i++)
        //    {
        //        tensors[i] = a;
        //    }
        //    return new MultiplicationTensor(tensors);
        //}
    }
}
