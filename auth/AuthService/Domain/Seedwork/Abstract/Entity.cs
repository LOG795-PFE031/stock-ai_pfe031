namespace AuthService.Domain.Seedwork.Abstract;

public abstract class Entity<T> : IEquatable<Entity<T>> where T : class
{
    public string Id { get; init; }

    protected Entity(string id)
    {
        if (string.IsNullOrWhiteSpace(id))
        {
            throw new ArgumentException("The ID cannot be null or whitespace.", nameof(id));
        }

        Id = id;
    }

    public bool Equals(Entity<T>? other)
    {
        if (ReferenceEquals(null, other)) return false;
        if (ReferenceEquals(this, other)) return true;
        return Id == other.Id;
    }

    public override bool Equals(object? obj)
    {
        return Equals(obj as Entity<T>);
    }

    public override int GetHashCode()
    {
        return Id.GetHashCode();
    }
}