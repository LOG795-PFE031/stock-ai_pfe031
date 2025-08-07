using MassTransit;

namespace RabbitMqNuget.Registration;

public sealed class MessageNameFormatter : IEntityNameFormatter
{
    public string FormatEntityName<T>()
    {
        return typeof(T).Name;
    }
}