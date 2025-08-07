using AuthService.Commands.Seedwork;
using AuthService.Monads;

namespace AuthService.Dispatchers;

public class CommandDispatcher(IServiceProvider serviceProvider, ILogger<CommandDispatcher> logger) : ICommandDispatcher
{
    public async Task<Result> DispatchAsync<TCommand>(TCommand command, CancellationToken cancellation) where TCommand : ICommand
    {
        try
        {
            var handler = serviceProvider.GetRequiredService<ICommandHandler<TCommand>>();

            if (command.GetType().Name is { } commandName && string.IsNullOrWhiteSpace(commandName) is false)
            {
                logger.LogTrace($"Dispatching Command '{commandName}'");
            }

            return await handler.Handle(command, cancellation)
                .FailedAsync(e =>
                {
                    logger.LogError(e, $"Error while handling command: '{command.GetType().Name}'");
                });
        }
        catch (Exception e)
        {
            logger.LogError(e, $"Error while handling command: '{command.GetType().Name}'");

            return Result.Failure(e);
        }
    }
}