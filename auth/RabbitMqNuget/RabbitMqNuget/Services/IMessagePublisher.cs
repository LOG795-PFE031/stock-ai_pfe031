namespace RabbitMqNuget.Services;

public interface IMessagePublisher<in TMessage> where TMessage : class
{
    Task Publish(TMessage message);
}