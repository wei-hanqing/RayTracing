import taichi as ti
import numpy as np
from ray_tracing_utils import Ray, Camera, Hittable_list, Sphere, PI, refract, reflect, reflectance, random_in_unit_sphere, random_unit_vector
# from ray_tracing_models import Ray, Camera, Hittable_list, Sphere, PI
ti.init(arch=ti.gpu)

@ti.func
def path_color(ray):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    p_RR = 0.8
    for n in range(10):
        if ti.random() > p_RR:
            break
        is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(
            Ray(scattered_origin, scattered_direction))
        if is_hit:
            if material == 0:
                color_buffer = color * brightness
                break
            else:
                # Diffuse
                if material == 1:
                    target = hit_point + hit_point_normal + random_unit_vector()
                    scattered_direction = target - hit_point
                    scattered_origin = hit_point
                    brightness *= color
                # Metal and Fuzz Metal
                elif material == 2 or material == 4:
                    fuzz = 0.0
                    if material == 4:
                        fuzz = 0.4
                    scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal) + \
                        fuzz * random_unit_vector()
                    scattered_origin = hit_point
                    if scattered_direction.dot(hit_point_normal) < 0:
                        break
                    else:
                        brightness *= color
                # Dielectric
                elif material == 3:
                    refraction_ratio = 1.5
                    if front_face:
                        refraction_ratio = 1 / refraction_ratio
                    cos_theta = min(1.0, -scattered_direction.normalized().dot(hit_point_normal))
                    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                    if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                        scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal)
                    else:
                        scattered_direction = refract(scattered_direction.normalized(), hit_point_normal, refraction_ratio)
                    scattered_origin = hit_point
                    brightness *= color
                brightness /= p_RR
    return color_buffer


@ti.func
def stack_clear(i, j):
    origin_stack_pointer[i, j] = -1
    direction_stack_pointer[i, j] = -1
    reflect_refract_stack_pointer[i, j] = -1
    color_weight_stack_pointer[i, j] = -1

@ti.func
def stack_push(i, j, hit_point, new_direction, color_weight):
    origin_stack_pointer[i, j] += 1
    direction_stack_pointer[i, j] += 1
    reflect_refract_stack_pointer[i, j] += 1
    color_weight_stack_pointer[i, j] += 1

    origin_stack[i, j, origin_stack_pointer[i, j]] = hit_point
    direction_stack[i, j, direction_stack_pointer[i, j]] = new_direction
    reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]] = ti.Vector([0, 0])
    color_weight_stack[i, j, color_weight_stack_pointer[i, j]] = color_weight

@ti.func
def stack_pop(i, j):
    origin_stack_pointer[i, j] -= 1
    direction_stack_pointer[i, j] -= 1
    reflect_refract_stack_pointer[i, j] -= 1
    color_weight_stack_pointer[i, j] -= 1

@ti.func
def stack_top(i, j):
    return origin_stack[i, j, origin_stack_pointer[i, j]], \
        direction_stack[i, j, direction_stack_pointer[i, j]], \
            reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]], \
                color_weight_stack[i, j, color_weight_stack_pointer[i, j]]

@ti.func
def blinn_phong(ray_direction, hit_point, hit_point_normal, color, material):
    hit_point_to_source = to_light_source(hit_point, light_source)
    # Diffuse
    brightness = ti.max(0.0, hit_point_to_source.dot(hit_point_normal) / hit_point_to_source.norm())
    diffuse_color = color * brightness

    specular_color = ti.Vector([0.0, 0.0, 0.0])
    diffuse_weight = 1.0
    specular_weight = 1.0
    # Specular
    if material != 1:
        H = (-(ray_direction.normalized()) + hit_point_to_source.normalized()).normalized()
        N_dot_H = ti.max(0.0, H.dot(hit_point_normal.normalized()))
        intensity = ti.pow(N_dot_H, 10)
        specular_color = intensity * color
    # Fuzz metal
    if material == 4:
        diffuse_weight = 0.5
        specular_weight = 0.5
    # Dielectric
    if material == 3:
        diffuse_weight = 0.1
    # Shadow
    shadow_weight = 1.0
    is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric = scene.hit_shadow(
        Ray(hit_point, hit_point_to_source))
    if not is_hit_source:
        if is_hitted_non_dielectric:
            # Hard shadow
            shadow_weight = 0
        elif hitted_dielectric_num > 0:
            # Soft shadow
            shadow_weight = ti.pow(0.5, hitted_dielectric_num)
    return (diffuse_weight * diffuse_color + specular_weight * specular_color) * shadow_weight

@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point

@ti.func
def ray_color(ray, i, j):
    default_color = ti.Vector([0.0, 0.0, 0.0])
    stack_clear(i, j)
    stack_push(i, j, ray.origin, ray.direction, 1.0)
    # scattered_origin = ray.origin
    # scattered_direction = ray.direction
    # # get color
    # is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(scattered_origin, scattered_direction))
    # if is_hit:
    #     if material == 0:
    #         default_color = color
    #     else:
    #         # hit_point_to_source = to_light_source(hit_point, ti.Vector([0, 5.4-3.0, -1]))
    #         # brightness = ti.max(0.0, hit_point_to_source.dot(hit_point_normal) / (hit_point_to_source.norm()))
    #         # default_color = color * brightness
    #         default_color = blinn_phong(scattered_direction, hit_point, hit_point_normal, color, material)
    # return default_color

    while origin_stack_pointer[i, j] >= 0 and origin_stack_pointer[i, j] < 10:
        curr_origin, curr_direction, curr_relect_refract, color_weight = stack_top(i, j)
        is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(curr_origin, curr_direction))
        if is_hit:
            # Light
            if material == 0:
                default_color = color * color_weight
                stack_pop(i, j)
            # Diffuse
            elif material == 1:
                local_color = blinn_phong(curr_direction, hit_point, hit_point_normal, color, material)
                default_color = local_color * color_weight
                stack_pop(i, j)
            # Metal
            elif material == 2 or material == 4:
                fuzz = 0.0
                if material == 4:
                    fuzz = 0.4
                refected = reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][0]
                if not refected:
                    reflected_direction = reflect(curr_direction.normalized(), hit_point_normal) + fuzz * random_in_unit_sphere()
                    reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][0] = 1
                    if reflected_direction.dot(hit_point_normal) > 0:
                        stack_push(i, j, hit_point, reflected_direction, 1.0)
                else:
                    local_color = blinn_phong(curr_direction, hit_point, hit_point_normal, color, material)
                    default_color += local_color
                    stack_pop(i, j)
            # Dielectric
            elif material == 3:
                refraction_ratio = 1.5
                if front_face:
                    refraction_ratio = 1 / refraction_ratio
                cos_theta = min(-curr_direction.normalized().dot(hit_point_normal), 1.0)
                sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                reflect_weight = reflectance(cos_theta, refraction_ratio)
                refract_weight = 1 - reflect_weight
                refected = reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][0]
                refracted = reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][1]
                if not refected:
                    reflected_direction = reflect(curr_direction.normalized(), hit_point_normal)
                    reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][0] = 1
                    stack_push(i, j, hit_point, reflected_direction, reflect_weight)
                else:
                    local_color = blinn_phong(curr_direction, hit_point, hit_point_normal, color, material)
                    default_color = default_color + 0.1 * local_color
                    stack_pop(i, j)
                if not refracted:
                    if refraction_ratio * sin_theta <= 1.0:
                        refracted_direction = refract(curr_direction.normalized(), hit_point_normal, refraction_ratio)
                        reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][1] = 1
                        stack_push(i, j, hit_point, refracted_direction, refract_weight)
                else:
                    local_color = blinn_phong(curr_direction, hit_point, hit_point_normal, color, material)
                    default_color = default_color + 0.1 * local_color
                    stack_pop(i, j)
            else:
                stack_pop(i, j)
        else:
            stack_pop(i, j)
    
    return default_color

@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / 960
        v = (j + ti.random()) / 960
        color = ti.Vector([0.0, 0.0, 0.0])
        brightness = 1.0
        for n in range(4):
            ray = camera.get_ray(u, v)
            # color += ray_color(ray)
            # color += ray_color(ray, i, j)
            color += path_color(ray)
        color /= 4
        canvas[i, j] += color

if __name__ == "__main__":
    canvas = ti.Vector.field(3, dtype=ti.f32, shape=(960, 960))
    light_source = ti.Vector([0, 5.4 - 3.0, -1])

    origin_stack = ti.Vector.field(3, dtype=float, shape=(960, 960, 10))
    origin_stack_pointer = ti.field(dtype=int, shape=(960, 960))
    direction_stack = ti.Vector.field(3, dtype=float, shape=(960, 960, 10))
    direction_stack_pointer = ti.field(dtype=int, shape=(960, 960))
    reflect_refract_stack = ti.Vector.field(2, dtype=int, shape=(960, 960, 10))
    reflect_refract_stack_pointer = ti.field(dtype=int, shape=(960, 960))
    color_weight_stack = ti.field(dtype=float, shape=(960, 960, 10))
    color_weight_stack_pointer = ti.field(dtype=int, shape=(960, 960))

    scene = Hittable_list()
    # Light source
    scene.add(Sphere(center=ti.Vector([0, 5.4, -1]), radius=3.0, material=0, color=ti.Vector([10.0, 10.0, 10.0])))
    # Ground
    scene.add(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # ceiling
    scene.add(Sphere(center=ti.Vector([0, 102.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # back wall
    scene.add(Sphere(center=ti.Vector([0, 1, 101]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # right wall
    scene.add(Sphere(center=ti.Vector([-101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])))
    # left wall
    scene.add(Sphere(center=ti.Vector([101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.0, 0.6, 0.0])))

    # Diffuse ball
    scene.add(Sphere(center=ti.Vector([0, -0.2, -1.5]), radius=0.3, material=1, color=ti.Vector([0.8, 0.3, 0.3])))
    # Metal ball
    scene.add(Sphere(center=ti.Vector([-0.8, 0.2, -1]), radius=0.7, material=2, color=ti.Vector([0.6, 0.8, 0.8])))
    # Glass ball
    scene.add(Sphere(center=ti.Vector([0.7, 0, -0.5]), radius=0.5, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    # Metal ball-2
    scene.add(Sphere(center=ti.Vector([0.6, -0.3, -2.0]), radius=0.2, material=4, color=ti.Vector([0.8, 0.6, 0.2])))


    camera = Camera()
    gui = ti.GUI("Ray Tracing", res=(960, 960))
    cnt = 0
    canvas.fill(0)
    while gui.running:
        render()
        cnt += 1
        gui.set_image((canvas.to_numpy() / cnt))
        gui.show()



