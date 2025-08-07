namespace RabbitMqNuget.Services.Impl;

public sealed class TransactionInfo : ITransactionInfo
{
    public Guid? CorrelationId { get; set; }
}