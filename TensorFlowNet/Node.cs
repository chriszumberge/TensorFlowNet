namespace TensorFlowNet
{
    public abstract class Node
    {
        public abstract string GraphText { get; }

        // TODO some type of evaluate graph children method so each node can control it's own output?.. get children?
    }
}
