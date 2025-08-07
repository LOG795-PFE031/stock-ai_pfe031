using MassTransit;

namespace RabbitMqNuget.Testing;

public sealed class ConsumerDecorator<TConsumed, TDecorated>(TDecorated decorated) : IConsumer<TConsumed>
    where TConsumed : class
    where TDecorated : class, IConsumer<TConsumed>
{
    public async Task Consume(ConsumeContext<TConsumed> context)
    {
        await decorated.Consume(context);

        if (context.CorrelationId is { } span && PublishExtensions.TransactionCompletedNotifier.TryGetValue(span, out var value))
        {
            value.SetResult(context.Message);

            PublishExtensions.TransactionCompletedNotifier.TryRemove(span, out _);
        }
    }
}