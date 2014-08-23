--[[
	ERL

	Pole balancing experiment - lua version

	Eric Laukien
]]--

-- Generate the field

local handle = generatePhenotype(16, 16, 3, 4, 1, 2, 2)

local pixelsPerMeter = 128.0
local poleLength = 1.0
local g = -2.8
local massMass = 20.0
local cartMass = 2.0
local massVel = { 0.0, 0.0 }
local poleAngle = 0.0
local poleAngleVel = 0.0
local poleAngleAccel = 0.0
local cartX = 0.0
local massPos = { cartX, poleLength }
local cartVelX = 0.0
local cartAccelX = 0.0
local poleRotationalFriction = 0.008
local cartMoveRadius = 1.8
local cartFriction = 0.02
local maxSpeed = 3.0

local dt = 0.017

local fitness = 0.0
local prevFitness = 0.0

local totalFitness = 0.0

for i = 1, 1200, 1 do
	-- Update fitness
	prevFitness = fitness

	if (poleAngle < math.pi) then
		fitness = -(math.pi * 0.5 - poleAngle)
	else
		fitness = -(math.pi * 0.5 - (math.pi * 2.0 - poleAngle))
	end

	fitness = fitness - math.abs(poleAngleVel * 0.25)

	totalFitness = totalFitness + fitness * 0.1

	--------------------------------- AI ---------------------------------

	local dFitness = fitness - prevFitness

	local err = dFitness * 10.0

	setPhenotypeInput(handle, 0, cartX * 0.25)
	setPhenotypeInput(handle, 1, cartVelX)
	setPhenotypeInput(handle, 2, math.fmod(poleAngle + math.pi, 2.0 * math.pi))
	setPhenotypeInput(handle, 3, poleAngleVel)

	stepPhenotype(handle, err, 8)

	local dir = math.min(1.0, math.max(-1.0, getPhenotypeOutput(handle, 0)))

	local agentForce = 4000.0 * dir

	------------------------------- Physics -------------------------------

	local pendulumCartAccelX = cartAccelX

	if (cartX < -cartMoveRadius) then
		pendulumCartAccelX = 0.0
	elseif (cartX > cartMoveRadius) then
		pendulumCartAccelX = 0.0
	end

	poleAngleAccel = pendulumCartAccelX * math.cos(poleAngle) + g * math.sin(poleAngle)
	poleAngleVel = poleAngleVel - poleRotationalFriction * poleAngleVel + poleAngleAccel * dt
	poleAngle = poleAngle + poleAngleVel * dt

	massPos = { cartX + math.cos(poleAngle + math.pi * 0.5) * poleLength, math.sin(poleAngle + math.pi * 0.5) * poleLength }

	local force = 0.0

	if (math.abs(cartVelX) < maxSpeed) then
		force = math.max(-4000.0, math.min(4000.0, agentForce))
	end

	if (cartX < -cartMoveRadius) then
		cartX = -cartMoveRadius

		cartAccelX = -cartVelX / dt
		cartVelX = -0.5 * cartVelX
	elseif (cartX > cartMoveRadius) then
		cartX = cartMoveRadius

		cartAccelX = -cartVelX / dt
		cartVelX = -0.5 * cartVelX
	end

	cartAccelX = 0.25 * (force + massMass * poleLength * poleAngleAccel * math.cos(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * math.sin(poleAngle)) / (massMass + cartMass)
	cartVelX = cartVelX - cartFriction * cartVelX + cartAccelX * dt
	cartX = cartX + cartVelX * dt

	poleAngle = math.fmod(poleAngle, 2.0 * math.pi)

	if (poleAngle < 0.0) then
		poleAngle = poleAngle + math.pi * 2.0
	end
end

-- Set experiment's fitness
setFitness(totalFitness)
