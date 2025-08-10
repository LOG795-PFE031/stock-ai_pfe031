namespace AuthService.Dtos;

public sealed class UserCredentials
{
    public required string Username { get; set; }
    public required string Password { get; set; }
}