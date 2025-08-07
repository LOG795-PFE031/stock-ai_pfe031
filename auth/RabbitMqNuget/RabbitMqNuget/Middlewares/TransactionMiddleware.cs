using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using RabbitMqNuget.Services;

namespace RabbitMqNuget.Middlewares;

public sealed class TransactionMiddleware
{
    private readonly RequestDelegate _next;

    public TransactionMiddleware(RequestDelegate next)
    {
        _next = next;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        var transactionInfo = context.RequestServices.GetRequiredService<ITransactionInfo>();

        if (context.Request.Headers.TryGetValue("CorrelationId", out var header))
        {
            transactionInfo.CorrelationId = Guid.Parse(header);
        }

        await _next(context);
    }
}
