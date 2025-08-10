namespace AuthService.Domain.Seedwork.Abstract;

public abstract class Aggregate<T> : Entity<T> where T : class
{
    public DequeueEnumerable<Event> DomainEvents { get; } = new();

    protected void AddDomainEvent(Event domainEvent)
    {
        DomainEvents.Enqueue(domainEvent);
    }

    protected Aggregate(string id) : base(id) {}
}