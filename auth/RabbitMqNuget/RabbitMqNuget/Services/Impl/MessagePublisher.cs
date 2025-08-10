using MassTransit;
using Microsoft.Extensions.Options;
using RabbitMqNuget.Registration;

namespace RabbitMqNuget.Services.Impl;

public sealed class MessagePublisher<TMessage> : IMessagePublisher<TMessage> where TMessage : class
{
    private readonly string _connectionString;
    private readonly ISendEndpointProvider _endpointProvider;
    private readonly ITransactionInfo _transactionInfo;

    public MessagePublisher(IOptions<RabbitMqOptions> settings, ISendEndpointProvider endpointProvider, ITransactionInfo transactionInfo)
    {
        _connectionString = settings.Value.Rabbitmq;
        _endpointProvider = endpointProvider;
        _transactionInfo = transactionInfo;
    }

    public async Task Publish(TMessage message)
    {
        Type messageConcreteType = message.GetType();

        var exchangeName = Registration.MassTransit.ExchangeNamesForMessageTypes[messageConcreteType];

        var endpoint = await _endpointProvider.GetSendEndpoint(new Uri($"{_connectionString}/{exchangeName}"));

        await endpoint.Send(message, messageConcreteType, context => context.CorrelationId = _transactionInfo.CorrelationId ?? Guid.NewGuid());
    }
}