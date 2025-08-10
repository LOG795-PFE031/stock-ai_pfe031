using System.Collections;

namespace AuthService.Domain.Seedwork;

public sealed class DequeueEnumerable<T> : IEnumerable<T>
{
    private readonly Queue<T> _queue = [];

    internal void Enqueue(T item) => _queue.Enqueue(item);

    public IEnumerator<T> GetEnumerator()
    {
        while (_queue.Count > 0)
        {
            yield return _queue.Dequeue();
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}