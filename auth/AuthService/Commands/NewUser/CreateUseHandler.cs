using AuthService.Commands.Seedwork;
using AuthService.Domain;
using AuthService.Monads;
using Microsoft.AspNetCore.Identity;
using RabbitMqNuget.Services;

namespace AuthService.Commands.NewUser;

public sealed class CreateUseHandler : ICommandHandler<CreateUser>
{
    private readonly UserManager<UserPrincipal> _userManager;
    private readonly IMessagePublisher<UserCreated> _messagePublisher;

    public CreateUseHandler(UserManager<UserPrincipal> userManager, IMessagePublisher<UserCreated> messagePublisher)
    {
        _userManager = userManager;
        _messagePublisher = messagePublisher;
    }

    public async Task<Result> Handle(CreateUser command, CancellationToken cancellation)
    {
        var identity = await _userManager.CreateAsync(new UserPrincipal(Guid.NewGuid().ToString())
        {
            UserName = command.Username,
            
        }, command.Password);

        if (!identity.Succeeded)
        {
            return Result.Failure(identity.Errors.First().Description);
        }

        var user = await _userManager.FindByNameAsync(command.Username);

        if (user == null)
        {
            return Result.Failure("User not found");
        }

        await _userManager.AddToRoleAsync(user, command.Role);

        await _messagePublisher.Publish(new UserCreated() {WalletId = user.WalletId});

        return Result.Success();
    }
}