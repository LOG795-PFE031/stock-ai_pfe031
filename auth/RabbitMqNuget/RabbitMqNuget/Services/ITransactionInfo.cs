namespace RabbitMqNuget.Services;

public interface ITransactionInfo
{
    public Guid? CorrelationId { get; set; }
}