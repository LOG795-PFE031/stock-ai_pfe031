using AuthService.Queries.Seedwork;

namespace AuthService.Queries.Jwt;

public sealed record GetJwtForCredentials(string Username, string Password) : IQuery;