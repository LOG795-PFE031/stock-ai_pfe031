using AuthService.Monads;

namespace AuthService.Commands.Seedwork;

public interface ICommandHandler<in TCommand> where TCommand : ICommand
{
    Task<Result> Handle(TCommand command, CancellationToken cancellation);
}