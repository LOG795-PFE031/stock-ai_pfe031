using RabbitMqNuget.Services;
using System.Collections.Concurrent;
using Microsoft.Extensions.DependencyInjection;

namespace RabbitMqNuget.Testing;

public static class PublishExtensions
{
    internal static readonly ConcurrentDictionary<Guid, TaskCompletionSource<object>> TransactionCompletedNotifier = new();

    public static async Task<TMessage> WithMessagePublished<TMessage>(this IServiceProvider serviceProvider, TMessage message, Guid correlationId = default) where TMessage : class
    {
        using var scope = serviceProvider.CreateScope();

        var transactionInfo = scope.ServiceProvider.GetRequiredService<ITransactionInfo>();

        transactionInfo.CorrelationId = correlationId = correlationId == default ? Guid.NewGuid() : correlationId;

        var tcs = new TaskCompletionSource<object>();

        TransactionCompletedNotifier.TryAdd(correlationId, tcs);

        var messagePublisher = scope.ServiceProvider.GetRequiredService<IMessagePublisher<TMessage>>();

        await messagePublisher.Publish(message);

        return (TMessage)await tcs.Task;
    }
}